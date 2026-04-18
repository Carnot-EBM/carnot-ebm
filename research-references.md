# Research References & Future Considerations

Items filed here are technologies, papers, repos, and ideas to consider
in future research milestones. The research conductor and planning agent
should read this file when designing new milestones.

## Inference Optimization

### RotorQuant — KV Cache Compression
- **Repo:** github.com/scrya-com/rotorquant
- **What:** Clifford algebra rotor-based KV cache compression for LLMs. 10.3x compression, better perplexity than Google TurboQuant, 28% faster decode, O(d) complexity via block-diagonal Givens/quaternion rotations.
- **Relevance:**
  1. Could enable running 1-3B+ models within same memory budget for the verify-repair pipeline (currently limited to 0.6-0.8B on CPU)
  2. Clifford algebra rotors (block-diagonal rotation math) could improve continuous Ising relaxation convergence
  3. llama.cpp integration path for lighter inference runtime in production
- **Tech:** Python + CUDA + Triton + Metal, targets llama.cpp
- **When to revisit:** When scaling local LLM size becomes a bottleneck, or exploring efficient inference for production

### HISA — Hierarchical Indexed Sparse Attention
- **Paper:** arxiv.org/abs/2603.28458
- **What:** Two-stage hierarchical sparse attention — block-level filtering then token-level refinement. Drop-in replacement for DeepSeek Sparse Attention, no retraining. Significant speedups at 64K+ context.
- **Relevance:** Not useful for current short-context verify-repair calls (<500 tokens). Becomes relevant when Carnot handles long-context verification (full codebases, multi-turn repair with conversation history, batch-packed examples).
- **When to revisit:** When adding long-document or codebase-level verification to the pipeline

## EBM Ecosystem

### Kona / Logical Intelligence (Primary Competitor and North Star)
- **Creator:** Eve Bodnia (PhD quantum information, UC Santa Barbara)
- **Architecture:** Non-autoregressive EBM reasoning in continuous latent space.
  Language-free. Maps answers onto energy landscapes (valleys = correct, peaks = wrong).
- **Results:** 76% Putnam benchmark, 96.2% Sudoku in 313ms, ~$4 vs ~$15K for complex reasoning.
- **Board:** Yann LeCun (founding Technical Research Board), Michael Freedman (Fields Medalist collaborator)
- **Key paper:** "Compression is all you need: Modeling Mathematics" (arXiv 2603.20396, March 2026)
  -- with Freedman. Models math compressibility via monoids, tested against Lean 4's MathLib.
  The monoid-based compressibility framework could inform how Carnot structures constraint
  spaces in Phase 3.
- **Blog:** logicalintelligence.com/blog/energy-based-models-for-reasoning
- **Talk:** "WTF is a Reasoning EBM?" (Neuron Daily interview with Eve Bodnia)
- **Relevance to Carnot Phase 3:**
  - Kona generates reasoning via energy minimization; Carnot currently verifies reasoning
  - Kona is continuous latent space; Carnot is discrete constraints (Ising/Z3)
  - Kona is language-free; Carnot is language-dependent extraction
  - Kona is closed-source; Carnot is Apache 2.0
  - The compression paper's monoid framework could bridge Carnot's discrete constraints
    to continuous energy landscapes
- **Action:** Read arXiv 2603.20396 in detail for Phase 3 architecture planning.
  Monitor Kona updates. Our unique advantage: open-source + hardware-portable.

### Extropic / thrml
- **Repo:** github.com/extropic-ai/thrml
- **What:** Thermodynamic sampling framework targeting Extropic TSU hardware
- **Relevance:** Carnot's parallel Ising sampler is 183x faster than thrml on CPU. TSU abstraction layer (Exp 71) is ready for when hardware ships.
- **Known issue:** thrml crashes on ROCm (extropic-ai/thrml#41)

### TorchEBM
- **What:** PyTorch-based EBM framework
- **Relevance:** Alternative approach to Carnot's JAX-based pipeline. Worth monitoring for ideas.

### EB-JEPA
- **What:** Energy-Based Joint Embedding Predictive Architecture
- **Relevance:** JEPA-style context prediction was explored early (Exp 1-38 era) but activation-based approaches proved insufficient. May become relevant again with continuous Ising (Exp 64+).

## Hardware

### FPGA Ising Machine — TSU Simulation Before Hardware
- **What:** Implement parallel Ising sampling in RTL on FPGA as a hardware
  stand-in for Extropic TSU. Each p-bit is a flip-flop with stochastic update
  based on neighbors and couplings. LFSR-based random number generation.
- **Why:** Test Carnot's hardware path before Z1 ships. FPGA gives true
  parallelism, custom bit-width, low latency, 10-100x better power than GPU.
- **Scale:** Small FPGA (Kria KV260, DE10-Nano) = 1k-10k p-bits (matches
  current experiment scale). Large FPGA (VU13P, Agilex) = up to 256k p-bits.
- **Integration:** Create `FpgaBackend` for SamplerBackend (Exp 71) that sends
  couplings over PCIe/AXI/USB and reads back sampled spins. Rest of pipeline
  stays in Python.
- **Prior art:** Tohoku University FPGA Ising machines, Microsoft Azure Quantum
  FPGA solver, Fujitsu Digital Annealer (commercial FPGA Ising machine).
- **Benchmark:** Compare FPGA vs CPU ParallelIsingSampler on Exp 46b (5000-var SAT).
- **What FPGA lacks vs TSU:** True thermal noise (FPGA uses pseudo-random LFSRs),
  analog-speed sampling (TSU = nanoseconds, FPGA = microseconds).
- **When to pursue:** When FPGA hardware is available for testing. Quickest path:
  1k-4k spin Verilog sampler with AXI-Lite interface + Python FpgaBackend wrapper.

### AMD ROCm on gfx1150 (Radeon 890M iGPU)
- JAX GPU backend crashes (HIP runtime assertion failure)
- gfx1100 emulation works but is 96x slower than CPU for matmul
- `JAX_PLATFORMS=cpu` is mandatory on this machine
- PyTorch ROCm works (3.3x speedup on Qwen3 inference)

## Autonomous Agent Frameworks

### AutoAgent — Declarative Agent Engineering
- **Repo:** github.com/kevinrgu/autoagent
- **What:** Meta-agent reads `program.md` (human-written goals), autonomously modifies `agent.py`, benchmarks via Harbor, hill-climbs on score. Overnight autonomous refinement.
- **Relevance:** Carnot's autoresearch is more sophisticated (three-gate + Ising, milestone planning, self-heal) but AutoAgent's `program.md` pattern is cleaner for expressing human intent. Borrowed this idea as `research-program.md`.
- **Borrowed:** Declarative intent document pattern → `research-program.md`

## Alternative Architectures for Constraint Verification

### Kolmogorov-Arnold Networks (KANs) — New Energy Tier (HIGH PRIORITY)
- **What:** Neural networks where edges have learnable nonlinear activation
  functions (parameterized splines) instead of fixed activations on nodes.
  Comparable accuracy to MLPs with a fraction of the parameters. Highly
  interpretable.
- **Relevance to Carnot:** The most natural next energy tier. Ising has fixed
  quadratic energy E = -s^T J s. A KAN-based energy function has learnable
  nonlinear energy E = sum of spline(s_i, s_j) over edges. Strictly more
  expressive than Ising while remaining interpretable. Fewer parameters than
  Gibbs MLP. Differentiable (splines have gradients) so slots directly into
  the Exp 66 differentiable pipeline.
- **Addresses:** The constraint learning ceiling from Exp 62/88 — linear Ising
  features can't capture nonlinear constraint relationships. KANs could learn
  what Ising misses with interpretable energy decomposition.
- **Model tier placement:** Ising (quadratic) → **KAN (spline)** → Gibbs (MLP)
  → Boltzmann (deep residual)
- **Hardware path:** Spline lookup tables are efficient in FPGA — potentially
  hardware-mappable like Ising.
- **When to pursue:** Next research milestone. Create `carnot-kan` energy tier.

### Liquid Neural Networks (LNNs) — Adaptive Constraints (HIGH PRIORITY)
- **What:** Continuous-time recurrent networks from MIT. Parameters adapt
  during inference via differential equations. Robust to noise and OOD data.
- **Relevance to Carnot:** Solves multi-turn agentic verification (Goal #2).
  A static Ising model can't adapt as an agent acts over time — new facts
  should change which constraints matter. An LNN-based constraint model
  updates its coupling strengths in response to new observations.
- **Also useful for:** Autoresearch constraint evaluation (adapt to current
  codebase state), noise-robust constraint extraction from adversarial or
  unusual LLM outputs (the Exp 88 failure mode).
- **When to pursue:** When agentic verification becomes the focus.

### Mamba / State Space Models — Constraint State Propagation
- **What:** Linear-complexity sequence models. Fixed-size state compression
  enables practically infinite context without KV cache VRAM spikes.
- **Relevance to Carnot:** Fixed-size constraint state for multi-step
  reasoning chains (Goal #2). Compress all verified facts from previous
  steps into a fixed vector. Also enables users to run larger LLMs locally
  (memory efficiency), helping with live model loading (Goal #1).
- **When to pursue:** When building multi-turn constraint propagation module.

### RWKV — Lightweight Constraint Propagation
- **What:** Trains like transformer, infers like RNN. Linear attention
  approximation. No KV cache. Active open-source ecosystem.
- **Relevance to Carnot:** Similar to Mamba but simpler and more
  community-driven. Good for edge deployment of constraint verification.
  Recursive inference (only needs previous hidden state) maps well to
  step-by-step constraint propagation.
- **When to pursue:** Alternative to Mamba for constraint state, especially
  for edge/embedded deployment.

### RetNet — Low Priority
- **What:** Multi-scale retention mechanism replacing attention. Parallel
  training + recurrent inference + chunkwise processing.
- **Relevance to Carnot:** No unique advantage for constraint verification.
  Training efficiency matters for foundation models, not small constraint
  models. Skip unless a specific need emerges.

## Papers & Concepts

### Apple GSM8K Adversarial Variant — LLMs Can't Do Math (HIGH PRIORITY)
- **Paper:** arxiv.org/pdf/2410.05229
- **What:** Apple researchers took GSM8K (grade-school math benchmark), made
  two changes: (1) swapped the numbers (same logic, different values), and
  (2) added one irrelevant sentence (e.g., "five of them were a bit smaller
  than average"). Models dropped up to 65%. Even o1-preview dropped from
  92.7% → 77.4%. 8-shot prompting didn't help.
- **Root cause:** LLMs pattern-match, not reason. They see "discount" →
  multiply, "smaller" → subtract, "inflation" → apply. Keyword scanning,
  not arithmetic. Changing only numbers in identical problems varies scores
  by 15 percentage points — benchmarks measure memory, not intelligence.
- **Relevance to Carnot:** THIS IS OUR THESIS. Carnot's constraint
  verification doesn't care about irrelevant sentences — it extracts the
  arithmetic and verifies independently. The verify-repair loop uses
  external verification (Ising energy), not more prompting.
- **Experiment needed:** Run Carnot's verify-repair pipeline on the Apple
  GSM8K adversarial variant. Show that:
  1. LLM accuracy drops (as Apple showed)
  2. Carnot's verify-repair maintains accuracy (because Ising catches the
     arithmetic errors regardless of irrelevant context)
  3. The improvement is LARGER on adversarial variants than standard GSM8K
     (because there are more errors to catch)
  This would be Carnot's most compelling result — maintaining accuracy on
  problems that break ALL other approaches including reasoning models.
- **When to pursue:** Next milestone. This is the credibility experiment.

### Exp 66: End-to-End Differentiable Constraint Reasoning (PRIORITY)
- **Source:** research-roadmap-v7.md Phase 8
- **What:** Full Kona-like pipeline, differentiable end-to-end:
  LLM generates logits → soft token probabilities → embedding →
  continuous Ising constraints (Exp 64) → energy →
  backpropagate energy gradient through constraints to logits →
  adjust LLM sampling distribution toward constraint-satisfying tokens.
- **Prerequisites:** Exp 64 (continuous relaxation ✅) + Exp 65 (embedding-space ✅) + live LLM (Exp 56 ✅)
- **Why it matters:** Moves from post-hoc verification to real-time energy-guided decoding. Constraints steer generation, not just verify after the fact. This is the path to Kona parity.
- **When:** Next research milestone after production shipping (2026.04.4 ✅)

### Continuous Self-Learning Architecture
- **Concept:** Carnot should get smarter with every query. Four tiers:
  online constraint weighting → persistent constraint memory → JEPA-style
  predictive verification → adaptive energy landscape structure.
- **Key finding from Exp 116:** LNN adaptation within a single chain hurts
  (10% vs 100% Ising). Adaptation must operate at the right timescale:
  static within chains, online across chains, persistent across sessions.
- **Hardware principle:** Every tier must have an acceleration path.
  Tier 1: CPU counters. Tier 2: FPGA pattern matching. Tier 3: GPU/NPU
  predictor. Tier 4: FPGA/TSU graph reconfiguration.
- **See research-program.md** "Continuous Self-Learning" section for full design.

### JEPA for Predictive Constraint Verification
- **Concept:** Joint-Embedding Predictive Architecture applied to constraints.
  Given partial LLM output (N tokens), predict constraint state of full output.
- **Why:** Current pipeline checks constraints AFTER generation. JEPA-style
  prediction enables checking BEFORE generation completes — steer in advance.
- **Implementation path:** Train encoder that maps partial responses to
  constraint energy space. The energy of the partial embedding predicts
  violations in the full response. Small model, trainable via CD on
  (partial_response, final_violation) pairs from accumulated verify-repair logs.
- **Hardware:** Predictor runs on GPU/NPU. If prediction says "high energy
  likely," trigger full Ising verification on FPGA/TSU. Otherwise skip.
  This creates a fast-path/slow-path architecture.

### KAN Adaptive Mesh Refinement for Energy Landscapes
- **Concept:** KAN splines naturally support adaptive complexity — add knots
  where the energy landscape is complex, remove where smooth. This is the
  Tier 4 "adaptive structure" mechanism.
- **Why:** Static KAN has fixed knot count per edge. Over time, some edges
  need more resolution (complex nonlinear constraints) while others can be
  simplified (nearly linear). Adaptive refinement learns WHERE to spend
  representational capacity.
- **Hardware:** Spline lookup tables in FPGA can be updated without full
  reconfiguration — just rewrite the LUT contents. Mesh refinement (adding/
  removing knots) requires partial FPGA reconfiguration.

### EBM Hallucination Detection via "Spilled Energy" (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2602.18671 — "LLM Hallucination Detection via Energy-Based Models" (ICLR 2026)
- **What:** Reinterprets autoregressive LLMs as EBMs via soft Bellman equation in max-entropy RL.
  Detects hallucinations via "spilled energy" — the discrepancy between logit energy (pre-softmax)
  and output energy (post-softmax). Factually incorrect outputs have higher spilled energy.
- **Relevance:** This is a conceptual bridge between Carnot's structural constraint verification
  and the LLM's internal energy signal. "Spilled energy" is detectable without external KB.
  Could add a fast factual-plausibility signal complementing Ising constraint verification.
- **When to pursue:** Next milestone. Add SpilledEnergyExtractor to pipeline as lightweight
  factual-plausibility check before KB-backed verification.

### FactNet — Billion-Scale Knowledge Graph for Verification
- **Paper:** arxiv.org/abs/2602.03417 — "FactNet: A Billion-Scale Knowledge Graph" (2026-02)
- **What:** 1.7B atomic assertions with 3.01B auditable evidence pointers; 92.1% grounding
  precision. Designed for factual claim verification. Open-source, structured as triples.
- **Relevance:** Could serve as the knowledge base for factual constraint extraction (Goal #3).
  Triples map directly to Carnot's ConstraintTerm protocol — each (subject, predicate, object)
  becomes an IsingConstraint on whether the LLM's output is consistent with FactNet.
- **When to pursue:** Factual extractor milestone. Use FactNet as KB source for factual claims.

### Energy-Based Transformers (EBTs) for Scalable Reasoning
- **Paper:** arxiv.org/abs/2507.02092 — "Energy-Based Transformers are Scalable Learners and Thinkers" (2025-07)
- **What:** Reformulate transformer prediction as energy minimization. 35% higher scaling rates
  than Transformer++ baseline on standard benchmarks. Energy function verifies compatibility
  between input and prediction.
- **Relevance:** Validates the EBM-for-verification architectural direction at scale. The
  "input-prediction compatibility" framing is exactly Carnot's constraint verification use case.
  Could inform a deeper Boltzmann-tier integration with LLM hidden states.
- **When to pursue:** Long-term. Consider for Boltzmann tier redesign in 6+ months.

### Quantum-Inspired FPGA Ising Machine with Sparsified Connectivity
- **Paper:** arxiv.org/abs/2604.04606 (2026-04)
- **What:** FPGA Ising machine using sparsified spin connectivity. 6x faster than simulated
  annealing; solves 1,600-spin problems vs 400-spin baseline. Uses quantum-inspired anneal schedule.
- **Relevance:** Directly applicable to Carnot's FPGA Ising backend (SamplerBackend Exp 71).
  Sparsification matches Carnot's sparse Ising work (Exp 61, clause-graph masking). The 6x
  speedup and 4x scale increase are directly transferable to FpgaBackend design.
- **When to pursue:** FPGA hardware milestone. Use this paper's sparsification and annealing
  schedule for the Kria KV260 FpgaBackend implementation.

### Hard-Constrained Neural Networks via Orthogonal Projection (Πnet)
- **Paper:** arxiv.org/abs/2508.10480 — "Hard-Constrained Neural Networks with Orthogonal Projections" (2025-08)
- **What:** Output layer using operator splitting to guarantee convex constraint satisfaction.
  The projection operator maps any unconstrained output onto the feasible constraint set.
- **Relevance:** Applicable to Carnot's continuous Ising relaxation (Exp 64) and gradient repair
  (Exp 87). The orthogonal projection idea is more principled than the current Langevin repair —
  would guarantee constraint satisfaction rather than just reduce energy. Could replace the
  "random restart" fallback in VerifyRepairPipeline.
- **When to pursue:** Repair pipeline improvement milestone. Add ProjectionRepair strategy
  alongside existing Langevin repair.

### ARM-EBM Bijection — Autoregressive LLMs Are Secretly EBMs
- **Paper:** arxiv.org/abs/2512.15605 (2025-12)
- **What:** Proves a formal bijection between autoregressive language models and energy-based models
  via the chain rule of probability. Every ARM implicitly defines an EBM over complete sequences.
  Derives soft Bellman equation connecting ARM generation to energy minimization.
- **Relevance:** Theoretical grounding for why constraint-based energy verification works. The ARM
  generates tokens that implicitly minimize an energy; Carnot's Ising sampler explicitly minimizes
  a complementary constraint energy. Together they form a dual verification architecture.
  Also the original source for the "spilled energy" concept (arXiv 2602.18671 extends this).
- **When to pursue:** Use as theoretical justification in milestone docs. Informs SpilledEnergy
  extractor design.

### SAVeR — Self-Auditing for Faithful Multi-Turn Reasoning
- **Paper:** arxiv.org/abs/2604.08401 (2026-04)
- **What:** Self-Auditing Verification and Repair framework enforces verification over agent beliefs
  with constraint-guided repair before action commitment. Two-turn structure: (1) agent proposes
  action, (2) auditor checks action against propagated constraint state, (3) if fails, repair and
  recommit. Achieves high faithfulness on multi-step reasoning benchmarks.
- **Relevance:** Direct implementation target for Carnot's multi-turn agentic verification goal
  (research-program.md Goal #4). SAVeR's "constraint state propagation" maps onto Carnot's
  ConstraintStateMachine (Exp 125). The repair-before-commit loop is Carnot's verify-repair
  loop applied inside the agent action step.
- **When to pursue:** Next milestone. Implement SAVeR-style two-turn verification wrapper around
  VerifyRepairPipeline for multi-turn agentic contexts.

### MathAgent — Constraint Graph Blueprints for Mathematical Reasoning
- **Paper:** arxiv.org/abs/2604.11188 (2026-04)
- **What:** Legislator-Executor paradigm where a "Legislator" agent synthesizes a constraint graph
  as a generation blueprint; an "Executor" agent generates text that must satisfy the graph.
  Constraint graph encodes: variable types, value ranges, logical dependencies between steps.
  Seed-free synthesis — no human-written examples needed.
- **Relevance:** The constraint graph structure is a formal version of what ConstraintTemplateLibrary
  (Exp 343) generates. MathAgent's "Legislator" is what Carnot's LLMExtractor should become:
  a small model that produces explicit constraint graphs from problem statements. The graph then
  drives Ising verification directly (constraint graph → Ising coupling matrix).
- **When to pursue:** LLM-as-extractor milestone. Implement MathAgent-style constraint graph
  generation as the output format of LLMExtractor.

### T-SKM-Net — Trainable Neural Constraint Satisfaction
- **Paper:** arxiv.org/abs/2512.10461 (2025-12)
- **What:** Neural framework integrating Sampling Kaczmarz-Motzkin (SKM) method as a differentiable
  layer for linear constraint satisfaction. Combines learned embeddings with algebraic projection
  onto constraint hyperplanes. Trains end-to-end; better than softmax-based constraint enforcement.
- **Relevance:** Alternative to Langevin MCMC for Carnot's repair step. SKM projection is more
  principled than random-walk repair — it steps directly toward the constraint boundary rather than
  exploring randomly. Could replace the current repair loop in VerifyRepairPipeline with guaranteed
  convergence in fewer steps.
- **When to pursue:** Repair pipeline improvement milestone. Compare SKM projection vs Langevin
  repair vs Πnet orthogonal projection (arXiv 2508.10480) on the same constraint satisfaction tasks.

### CIKAN — Constraint-Informed Kolmogorov-Arnold Networks
- **Paper:** arxiv.org/abs/2412.03710 (2024-12, CIKAN)
- **What:** Extends KAN with hard physics/constraint priors embedded directly into spline activations.
  Constraint-informed splines have their knot placement and activation shapes shaped by known constraint
  boundaries — any output that violates constraints has infinite energy in the constrained subspace.
  Applied to autonomous spacecraft rendezvous with safety constraints; eliminates constraint violation
  entirely without post-hoc repair.
- **Relevance:** Carnot's KAN energy tier (Exp 96) uses standard KANs. CIKAN adds hard constraint
  priors directly into the spline structure: constraints become embedded in the energy landscape's
  topology rather than being checked after the fact. This would be the "Tier 4: Adaptive Structure"
  mechanism — the KAN's spline edges encode which constraint subspaces are valid.
- **Concrete experiment:** CIKANEnergy subclass of KANEnergy where each spline activation is seeded
  with known constraint boundaries (e.g., carry-check, range-check). Violation energy is infinite
  at the constraint boundary. Compare CIKAN vs standard KAN on constraint satisfaction tasks.
- **When to pursue:** Next milestone. Implement CIKANEnergy as new energy tier subclass.

### Digitally Optimized Initializations for Fast Thermodynamic Computing
- **Paper:** arxiv.org/abs/2603.24183 (2026-03)
- **What:** Shows that clever digital initialization of thermodynamic (analog) computing substrates
  dramatically accelerates convergence — analogous to warm-starting Ising annealing. Presents a
  framework for finding good initial spin configurations via classical preprocessing, then handing off
  to the thermodynamic substrate for final low-energy resolution. Reports 3-10x speedup on real
  thermodynamic hardware with optimal initialization.
- **Relevance:** Carnot's FpgaBackend (Exp 289) and future TSU integration (Exp 71 abstraction) can
  use this pattern: run a fast classical greedy initialization, pass the partially-solved spin
  configuration to FPGA/TSU for final annealing. This hybridizes CPU preprocessing with hardware
  sampling. The CPU preprocessor is the digital optimizer; FPGA/TSU does final minimization.
- **When to pursue:** FPGA hardware milestone. Add WarmStartSchedule to FpgaBackend that computes
  initial spin assignments via greedy descent before submitting to hardware.

### RLVR — Reinforcement Learning with Verifiable Rewards for Reasoning
- **Paper:** arxiv.org/abs/2506.14245 (2025-06)
- **What:** Shows that RLVR (RL with verifiable reward signals) on base LLMs implicitly teaches
  correct reasoning structure, not just answer accuracy. The verifiable reward acts as an external
  energy signal that guides the policy away from invalid reasoning chains. Base models (no SFT)
  trained on RLVR outperform instruction-tuned models on reasoning benchmarks without reward hacking.
- **Relevance:** Carnot's SelfLearningRelay (Exp 361) currently only updates constraint weights
  (Tier 1) and activates templates (Tier 2). RLVR suggests a Tier 3 path: use EORM energy scores
  as verifiable rewards to fine-tune the pipeline's constraint generation policy. The EORM model
  IS the verifiable reward signal. This connects Carnot's energy model directly to the LLM training loop.
- **When to pursue:** Self-learning milestone. Experiment with EORM-as-reward for online fine-tuning
  of constraint extraction policy (LLMExtractor). The LLMExtractor generates constraint graphs;
  EORM scores whether the constrained output is better; gradient flows back to extractor.

### Semantic Energy — Detecting LLM Hallucination Beyond Entropy (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2508.14496 (2025-08, updated 2025-12)
- **What:** Introduces Semantic Energy, a hallucination detection framework that operates directly
  on penultimate-layer logits (pre-softmax) using a Boltzmann-inspired energy distribution.
  Combines semantic clustering with energy scoring: cluster semantically equivalent outputs, then
  compute energy of each cluster. Higher energy = more hallucination-prone generation.
  Consistently outperforms semantic entropy on uncertainty estimation benchmarks.
- **Relevance:** Direct energy-based signal complementing Carnot's constraint verification.
  Where Ising verifies logical constraints AFTER generation, Semantic Energy scores the generation's
  reliability DURING output via logit intensity. Together they form a dual-signal pipeline:
  Semantic Energy for quick plausibility gating, Ising for structural constraint verification.
  The "logit intensity lost during softmax" insight directly applies to Carnot's SpilledEnergy
  concept (arXiv 2602.18671) — both tap the same pre-softmax signal.
- **Concrete experiment:** SemanticEnergyScorer class wrapping the penultimate-layer logit energy.
  Integrate as a fast-path pre-filter in VerifyRepairPipeline: high semantic energy → trigger full
  Ising verification; low semantic energy → skip. Compare vs SinkProbe for skip rate and FN rate.
- **When to pursue:** Next milestone. Implement SemanticEnergyScorer and benchmark vs SinkProbe.

### CRANE — Constrained Reasoning via Alternating Decoding (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2502.09061 (2025-02, ICLR 2026 accepted)
- **What:** CRANE (Constrained Reasoning with Alternating N-gram Extension) solves the fundamental
  tension between constrained generation (enforces structure, kills reasoning) and free generation
  (allows reasoning, loses structure). Key insight: augment the output grammar with "reasoning
  escape" rules that allow unconstrained text until a trigger token, then re-enter constrained
  mode for the final answer. Results: up to +10pp on GSM-symbolic and FOLIO vs both constrained
  and unconstrained baselines.
- **Relevance:** Directly applicable to Carnot's constraint extraction problem on IT models.
  The root cause of ArithmeticExtractor's 0% detection on Gemma4-E4B-it is that IT models reason
  freely and produce claims in natural language format, not regex-parseable equations. CRANE
  suggests the fix: instead of post-hoc extraction, add a constrained extraction suffix to the
  prompt grammar — let the model reason freely (IT format), then force a structured CLAIM: block
  at the end (constrained format). This eliminates the free-form extraction problem entirely.
- **Concrete experiment:** CRANEExtractionGate class that appends a constrained suffix prompt
  ("State all verifiable numerical claims as: CLAIM: [quantity] [op] [quantity] = [result]") and
  parses the structured suffix. Compare detection rate vs LLMExtractor and ArithmeticExtractor.
- **When to pursue:** Next milestone. Implement as alternative extraction front-end.

### Differentiable Symbolic Planning with Feasibility Channels (DSP)
- **Paper:** arxiv.org/abs/2604.02350 (2026-04)
- **What:** DSP is a neural architecture that performs discrete symbolic constraint reasoning while
  remaining fully differentiable end-to-end. Key component: "feasibility channel" (phi) that tracks
  constraint satisfaction evidence at each reasoning node and aggregates into a global feasibility
  signal via learned rule-weighted combination. Integrated into Universal Cognitive Kernel (UCK)
  with graph attention + iterative constraint propagation.
  Results: 97.4% on planning feasibility under 4x size generalization, 96.4% on SAT.
- **Relevance:** The feasibility channel concept maps directly onto Carnot's multi-step constraint
  propagation (SAVeR, ConstraintStateMachine). Instead of binary constraint state per step, DSP
  tracks a continuous feasibility signal that degrades smoothly as constraints are violated.
  This is the Tier 2 constraint memory architecture done right: phi encodes accumulated constraint
  satisfaction evidence across steps, feeds into the next step's prior. DSP's "rule-weighted
  combination" is the differentiable version of Carnot's template wiring (Exp 343/344).
- **When to pursue:** SAVeR/multi-turn milestone. Replace binary ConstraintState with continuous
  feasibility channel from DSP. Enables soft constraint propagation across reasoning steps.

### Restoring Sparsity in Potts Machines via Mean-Field Constraints
- **Paper:** arxiv.org/abs/2602.04200 (2026-02)
- **What:** Addresses constraint-induced graph density in Potts machines (multi-state Ising
  machines). Dense pairwise constraint couplings make hardware embedding NP-hard. Solution:
  Mean-Field Constraints (MFC) replaces dense pairwise couplings with dynamically updated
  single-node biases, achieving comparable solution quality while maintaining sparsity.
  Validates p-dit (probabilistic digit) dynamics with nearest-neighbor selection; proves
  detailed balance for correct stationary distribution.
- **Relevance:** Carnot's FpgaBackend (Exp 289) and CIKAN energy tier both face the density
  problem — adding constraint boundaries increases coupling density, making FPGA embedding harder.
  MFC's technique of replacing pairwise couplings with single-node bias updates is directly
  applicable: encode constraint satisfaction as bias corrections rather than additional couplings.
  This keeps the Ising graph sparse (FPGA-friendly) while preserving constraint semantics.
- **When to pursue:** FPGA hardware milestone. Add MFC bias correction to FpgaBackend to
  maintain sparse coupling structure when constraint boundaries are added.

### LLM-QUBO: LLMs as Constraint-to-QUBO Translators
- **Paper:** arxiv.org/abs/2509.00099 (2025-08)
- **What:** End-to-end framework where an LLM translates natural language problem descriptions
  into QUBO format for quantum annealing. The LLM acts as an expert rule-based conversion engine:
  identifies variables, objectives, constraints, then synthesizes a Python QUBO class separating
  objective function from constraint penalty terms. Handles hybrid quantum-classical decomposition
  (Benders' decomposition) for problems exceeding quantum hardware capacity.
- **Relevance:** Exact same goal as Carnot's LLMExtractor, but targeting quantum hardware.
  LLM-QUBO's NL-to-constraint-graph pipeline is the extraction pipeline Carnot needs:
  natural language problem → formal constraint representation. The output format (QUBO Python class)
  is structurally equivalent to Carnot's ConstraintTerm protocol. LLM-QUBO's template for
  separating objective from penalty is a clean pattern for LLMExtractor's output format.
  Could use LLM-QUBO's prompt templates as starting point for LLMExtractor's constraint
  graph generation.
- **When to pursue:** LLMExtractor improvement milestone. Study LLM-QUBO's prompt engineering
  for structured constraint extraction. Adapt for Carnot's ConstraintTerm output format.

### EBM-CoT — Energy-Based Calibration for Implicit Chain-of-Thought
- **Paper:** arxiv.org/abs/2511.07124 (2025-11)
- **What:** Introduces EBM-CoT, which uses an EBM to steer latent chain-of-thought representations
  toward lower-energy, higher-consistency regions without modifying the base LLM. The EBM is
  applied in the embedding space of CoT steps — low energy = high consistency across reasoning
  steps. Improves multi-step accuracy on math, commonsense, and symbolic reasoning benchmarks.
- **Relevance:** Carnot's JEPA predictive verification (Tier 3) operates in embedding space over
  CoT steps. EBM-CoT validates this approach with a working implementation. Their "consistency
  energy" over CoT steps is exactly what Carnot's JEPA predictor should learn: given partial
  CoT, predict whether the completed chain will be energy-consistent. Architecture details
  map onto Carnot's EORM + JEPA integration.
  Also used in Exp 291: isotonic calibration (arXiv 2511.07124) for JEPA predictor.
- **When to pursue:** EORM retrain milestone. Use EBM-CoT's consistency energy formulation
  as training objective for EORM and JEPA predictor.

### HalluField — Field-Theoretic Hallucination Detection
- **Paper:** arxiv.org/abs/2509.10753 (2025-09)
- **What:** Models each LLM response as a collection of token paths with associated energy and
  entropy values. Hallucinations appear as anomalous energy/entropy distributions in the token-path
  "field." Training-free, operates directly on logits. Provides a field-theoretic vocabulary
  that complements statistical mechanics approaches.
- **Relevance:** Carnot's energy tiers (Ising/KAN/Boltzmann) compute scalar energy over
  configurations. HalluField extends this to token-path distributions — a natural generalization.
  The "field" view enables per-token energy attribution: which specific tokens contributed most
  to a high-energy (hallucination-prone) output. This is useful for targeted repair — fix the
  high-energy token, not the whole response.
- **When to pursue:** SpilledEnergy + Semantic Energy milestone. Study HalluField's per-token
  energy attribution as a repair targeting mechanism.

### Hallucination Basins — Attractor Geometry in Latent Space
- **Paper:** arxiv.org/abs/2604.04743 (2026-04)
- **What:** Frames LLM hallucinations as attractor basins in latent space with geometry-dependent
  stability. Introduces steering techniques based on basin curvature and volume. High-curvature
  basins = fragile knowledge. Low-curvature basins = stable hallucinations. Basin volume
  correlates with hallucination frequency.
- **Relevance:** Carnot's Ising/KAN energy landscape IS an attractor basin map. Low energy =
  valid configuration basin. High energy = constraint-violating basin. The basin curvature
  insight maps onto KAN spline curvature — sharper splines at constraint boundaries = more
  stable structural constraint enforcement. Suggests using basin volume as a confidence signal:
  large valid-basin = high confidence, small valid-basin = verify more aggressively.
- **When to pursue:** KAN adaptive mesh refinement milestone (Tier 4). Use basin volume
  estimation to guide KAN spline refinement — add knots in small/fragile basins.

### Self-Adaptive Ising Machine — Lagrange Constraint Relaxation
- **Paper:** arxiv.org/abs/2501.04971 (2025-01)
- **What:** Proposes a self-adaptive Ising machine that shapes its energy landscape via Lagrange
  relaxation of constraints without manual penalty tuning. Constraint violations are penalized
  automatically via dual variable updates that track feasibility. Outperforms Fujitsu Digital
  Annealer on 300-variable knapsack problems. No manual lambda tuning required.
- **Relevance:** Carnot's constraint satisfaction layer currently uses fixed penalty weights.
  Self-adaptive Ising's Lagrange relaxation would automate this: the dual variables are the
  constraint weights that Carnot's Tier 1 online learning currently maintains manually.
  This is the hardware-compatible version of Tier 1 self-learning — the Ising machine ITSELF
  learns constraint weights via dual variable updates, at sampling speed.
- **When to pursue:** FPGA hardware milestone. Add LagrangeConstraintSchedule to FpgaBackend
  that auto-tunes penalty weights via dual variable updates. Bridges Tier 1 learning to hardware.

### ML-Assisted Dynamic Ising — Automated Parameter Selection
- **Paper:** arxiv.org/abs/2503.23966 (2025-03)
- **What:** Uses machine learning to automate parameter selection for simulated bifurcation Ising
  machines, enabling high-speed solving of dynamically changing combinatorial problems without
  manual parameter tuning. ML model predicts optimal beta schedule and coupling strengths from
  problem structure. Handles problems where constraint sets change between calls.
- **Relevance:** Carnot's verify-repair pipeline uses Ising sampling per constraint check.
  Each check has slightly different problem structure (different constraint graph, different
  variable count). ML-assisted parameter selection would automatically tune the annealing
  schedule and coupling scaling per problem — no fixed beta schedule needed. This is the
  per-query self-adapting that Tier 1 currently approximates with PerModelFPTracker.
- **When to pursue:** Ising optimization milestone. Add ML-guided annealing schedule to
  ParallelIsingSampler and FpgaBackend.

### Generative Thermodynamic Computing
- **Paper:** arxiv.org/abs/2506.15121 (2025-06, revised 2025-10)
- **What:** Demonstrates that Langevin dynamics can generate structured data from noise without
  neural networks or injected pseudorandom noise. The physical system's natural time evolution
  (thermal fluctuations) performs generation via the fluctuation-dissipation theorem. If realized
  in analog hardware, generation requires no random number generator, no model weights, and
  no gradient computation — just thermodynamic equilibration.
- **Relevance:** This is the core physics behind Extropic's TSU and Carnot's long-term Phase 2
  hardware path. It proves that sampling (the core operation in Carnot's Ising/Boltzmann tiers)
  is a physically native operation in thermodynamic hardware — not a simulation. The "no
  pseudorandom noise" result means TSU-style hardware would be inherently more efficient than
  FPGA Ising (which uses LFSRs). Validates the TSU hardware investment.
- **When to pursue:** Phase 2 hardware planning. Use as theoretical justification when designing
  TSU integration experiments. Also relevant to FPGA bitfile design: minimize pseudorandom
  requirements in LFSR to approximate thermodynamic sampling more faithfully.

### LLM-JEPA — Language Models Meet Joint Embedding Predictive Architectures
- **Paper:** arxiv.org/abs/2509.14252 (2025-09) — LeCun group (Huang, LeCun, Balestriero)
- **What:** Adapts JEPA (joint embedding predictive architecture, Yann LeCun's framework) for
  language. Shows that embedding-space training objectives are superior to input-space (token-level)
  prediction for representation quality and downstream reasoning. The JEPA energy in embedding
  space captures high-level semantic consistency that token-level cross-entropy misses.
- **Relevance:** Carnot's JEPA-style predictive verification operates in embedding space,
  which this paper validates as the RIGHT design choice. The paper provides architecture details
  for the JEPA encoder that Carnot's violation predictor should use. LeCun's group publishing
  this validates the JEPA verification approach at scale — if they're adapting JEPA to language,
  constraint energy in embedding space is the right architecture for verification.
  Also: direct connection to Kona (LeCun is on Kona's technical board) — this is the
  architectural direction Kona is taking.
- **When to pursue:** JEPA retrain milestone. Use LLM-JEPA encoder architecture for violation
  predictor. Compare embedding-space vs token-space verification accuracy.

(Add more papers, arxiv links, and theoretical ideas here as they come up)

### EORM — Energy-Based Outcome Reward Model for CoT Ranking (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2505.14999 (2025-05)
- **What:** Trains a 55M-parameter energy-based reward model that ranks chain-of-thought solutions
  by correctness without binary labels. Uses contrastive training on (correct, incorrect) CoT pairs.
  Small model size — 55M params — outperforms larger discriminative reward models on math reasoning.
- **Relevance:** This IS Carnot's Tier 3 JEPA predictive verification, spelled out. Instead of
  predicting constraint violations from partial responses, EORM ranks full CoT solutions by energy.
  The 55M-parameter scale fits on GPU alongside the LLM. Architecture: encode full CoT → scalar
  energy → rank by energy → select lowest-energy response. Replace or augment JEPA gate with EORM.
- **When to pursue:** Next milestone. Train EORM on accumulated live benchmark (CoT, correctness)
  pairs from Exp 340. Compare against JEPA gate accuracy on same held-out data.

### SinkProbe — Attention Sinks as Hallucination Detection Signal
- **Paper:** arxiv.org/abs/2604.10697 (2026-04)
- **What:** Hallucination detection via attention sink analysis — specific attention head activations
  cluster at "sink" tokens (like [BOS]) in a way that correlates with factual confidence.
  Model-agnostic, no auxiliary model needed. Low latency (single forward pass).
- **Relevance:** Complementary detection signal to Carnot's energy-based verification. SinkProbe
  fires on factually-uncertain outputs; Carnot's Ising/Z3 checks fire on structurally-wrong outputs.
  Ensemble of SinkProbe + SpilledEnergy + Ising gives multi-signal coverage across error types.
  SinkProbe is fast (no sampling), making it a good pre-filter before expensive Ising verification.
- **When to pursue:** Next milestone. Add SinkProbe as a fast pre-filter in the pipeline.

### Eidoku — Neuro-Symbolic CSP Verification Gate for LLM Reasoning
- **Paper:** arxiv.org/abs/2512.20664 (2025-12)
- **What:** Reformulates LLM reasoning verification as a Constraint Satisfaction Problem (CSP).
  Structural constraints (type consistency, value range, logical dependency) are verified via a
  neural-symbolic gate before accepting LLM outputs. Achieves near-perfect accuracy on structured
  reasoning tasks.
- **Relevance:** Validates Carnot's approach of expressing verification as constraint satisfaction.
  The "structural constraint" layer (type consistency, value range) maps onto Carnot's
  ArithmeticExtractor + NL2Z3 pipeline. Eidoku's gate design could improve the CoT circuit
  verifier (Exp 336) with more principled structural constraint types.
- **When to pursue:** Constraint extraction improvement milestone. Study Eidoku's constraint type
  taxonomy for new constraint categories to add to ConstraintTemplateLibrary.

### LLM-Guided Quantified SMT Solving
- **Paper:** arxiv.org/abs/2601.04675 (2026-01)
- **What:** Uses LLM guidance to improve Z3 performance on quantified SMT problems. Achieves 80%
  improvement in Z3 and 183.6% improvement in CVC5. LLM proposes instantiation candidates
  that guide quantifier elimination in the solver.
- **Relevance:** Reverse complement to Carnot's NL2Z3 approach (LLM→Z3 instead of Z3→repair).
  Suggests a bidirectional pipeline: Carnot extracts Z3 constraints from LLM responses, while
  this technique uses LLMs to help Z3 solve those constraints faster. Could speed up Z3 verification
  on complex quantified arithmetic constraints that NL2Z3 generates.
- **When to pursue:** Z3 extraction + performance milestone. Combine with VERGE loop.

### Energy-Guided Decoding for VLM Object Hallucination Mitigation
- **Paper:** arxiv.org/abs/2507.07731 (2025-07)
- **What:** Energy-based layer selection during decoding reduces object hallucinations in vision-language
  models. Energy function computed over token generation steps — high-energy transitions trigger
  guided decoding to steer away from hallucination. Plug-and-play, no retraining.
- **Relevance:** Most direct published application of energy-guided decoding to hallucination
  mitigation. The token-level energy computation maps onto Carnot's guided decoding goal (FR-12).
  The layer selection insight — energy guides which hidden layer drives token probabilities — is
  directly applicable to Carnot's LLM integration. Study as implementation blueprint for the
  guided decoding adapter.
- **When to pursue:** Guided decoding milestone. Use layer-selection insight for token probability adjustment.

### Scalable Connectivity for Ising Machines: Dense to Sparse
- **Paper:** arxiv.org/abs/2503.01177 (2026-03)
- **What:** Systematic study of sparse connectivity in FPGA Ising machines. Analyzes how connectivity
  reduction (dense → sparse) affects solution quality and hardware efficiency. Identifies which
  constraint graph topologies are robust to sparsification and which require dense coupling.
- **Relevance:** Directly applicable to Carnot's FPGA Ising backend design (FpgaBackend, SamplerBackend).
  Carnot's sparse Ising work (Exp 61, clause-graph masking) already uses sparse coupling. This paper
  provides principled guidelines for choosing sparsity level for the KV260 implementation —
  specifically whether Carnot's constraint graphs can be solved at 4x reduced connectivity.
- **When to pursue:** KV260 FPGA hardware milestone. Use to choose coupling sparsity for bitfile.

### Neural Uncertainty Principle — Prefill-Stage Hallucination Detection (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2603.19562 (2026-03)
- **What:** Adversarial vulnerability and hallucination share a geometric origin — input and loss-gradient are conjugate observables with an irreducible uncertainty bound (analogous to Heisenberg's principle). A prefill-stage probe detects hallucination risk BEFORE any tokens are generated, using only the input representation — no decoding required. ConjMask and LogitReg are actionable techniques.
- **Relevance:** Strongest new theoretical paper for Carnot. The prefill-stage energy probe maps directly onto Carnot's Ising/EBM scoring pipeline. Gives theoretical grounding for why energy scores predict hallucination. Could serve as the fast-path gate before the expensive Ising verification: if the prefill uncertainty is below threshold, skip full verification.
- **When to pursue:** Next milestone. Add PrefillUncertaintyProbe to pipeline. Pairs with SpilledEnergyExtractor (post-generation) for full coverage — pre-generation + post-generation.

### LogitScope — Varentropy-Based Hallucination Detection
- **Paper:** arxiv.org/abs/2603.24929 (2026-03)
- **What:** Token-level entropy and varentropy computed from logit distributions identify hallucination-prone decision points without labeled data. Model-agnostic, works with any HuggingFace model. Varentropy measures variance of entropy across tokens — high varentropy = high uncertainty about uncertainty = likely hallucination.
- **Relevance:** Directly operationalizable as a lightweight Carnot probe layer. Varentropy complements Spilled Energy (2602.18671) and Semantic Energy (2508.14496) — forms a three-signal extraction-free detection ensemble. Include as baseline comparison in upcoming extraction-free experiments.
- **When to pursue:** Next milestone. Add varentropy signal to SemanticEnergyExtractor module.

### SciDC — Multi-Layer Formal Constraints for LLM Decoding (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2604.06603 (2026-04)
- **What:** Converts domain knowledge into multi-layered formal rules that constrain LLM decoding at generation time. +12% accuracy on scientific tasks (retrosynthesis, clinical diagnosis). Multi-layer rule hierarchy: hard constraints (must satisfy), soft constraints (prefer), and domain-specific axioms.
- **Relevance:** Demonstrates energy-guided constrained decoding at production scale in hard scientific domains. The multi-layer rule hierarchy maps directly onto Carnot's constraint satisfaction tiers (Ising for hard, KAN for soft, Gibbs for domain-specific). This is the closest published system to Carnot's Goal #6 (energy-guided decoding). Study the rule hierarchy design for the guided decoding adapter.
- **When to pursue:** Guided decoding milestone. Borrow the multi-layer constraint hierarchy design.

### Talking with Verifiers — Auto-Generate Z3 Specs from NL (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2603.02235 (2026-02)
- **What:** Natural language requirements are automatically translated into formal verification queries compatible with existing NN verifiers. Bridges human-readable ↔ machine-verifiable gap. Auto-generates Z3/SMT specs from natural language descriptions.
- **Relevance:** Directly addresses Carnot's constraint extraction bottleneck. Auto-generating Z3/SMT specs from natural language is the missing front-end for the NSVIF pipeline (2601.17789). Could replace the manual constraint specification step entirely — user describes constraints in NL, system generates Z3 specs automatically.
- **When to pursue:** Z3 extraction milestone. Use as front-end for FormalClaimVerifier.

### Digitally Optimized Thermodynamic Initializations
- **Paper:** arxiv.org/abs/2603.24183 (2026-03)
- **What:** Hybrid digital-thermodynamic algorithm suppresses slow relaxation modes via Mpemba-effect-inspired initialization, yielding analytic speedups for matrix inversion on thermodynamic hardware.
- **Relevance:** Applicable to Carnot's FPGA/thermodynamic hardware path. Fast thermalization of the Ising sampler is a known bottleneck (slow mixing chains). Mpemba-effect initialization could reduce the burn-in period for FpgaBackend from O(n²) to O(n log n). Directly applicable to the KV260 Verilog sampler design.
- **When to pursue:** FPGA hardware milestone. Incorporate into FpgaBackend annealing schedule.

### Predicting Sampling Advantage of Stochastic Ising Machines
- **Paper:** arxiv.org/abs/2504.18359 (2026-04)
- **What:** Analyzes computational advantage regimes for stochastic Ising machines vs. GPU-based MCMC; identifies problem classes where Ising hardware wins. Defines a "hardness metric" for constraint graphs that predicts whether the Ising machine will outperform GPU sampling.
- **Relevance:** Grounds the Carnot hardware roadmap. Defines which constraint graphs benefit from Ising accelerator vs. GPU sampling, informing FPGA tier selection strategy. Can be used to decide when to route a query to FpgaBackend vs CPU/GPU sampler.
- **When to pursue:** FPGA benchmark milestone. Use hardness metric to predict routing advantage.

### LLM-JEPA — Joint Embedding Predictive Architecture for Language Models
- **Paper:** arxiv.org/abs/2509.14252 (2025-09)
- **What:** JEPA-based architecture applicable to both finetuning and pretraining of LLMs. Outperforms standard LLM training objectives with robustness to overfitting. Predicts future token embeddings from context using energy minimization.
- **Relevance:** Direct architecture for Carnot's Tier 3 JEPA predictive verification. Training the predictor on (partial_response, final_violation) pairs can borrow the LLM-JEPA training objective. Also provides a path to train the JEPA predictor ON TOP of an existing LLM embedding layer rather than from scratch.
- **When to pursue:** JEPA real-data training milestone (Milestone 2026.04.23 Phase 1).

### Emergent Formal Verification — Z3 SMT for Multi-Domain AI Safety
- **Paper:** arxiv.org/abs/2603.21149 (2026-03)
- **What:** Autonomous AI ecosystem independently discovers Z3 SMT-based verification across 6 safety domains (code, API safety, reasoning correctness, CLI validation, hardware, smart contracts). Achieves 100% accuracy with zero false positives by formalizing natural language requirements as SMT assertions.
- **Relevance:** Validates the NSVIF/Z3 approach for constraint extraction (research-program.md Goal #1b). The "NL → Z3 spec" pipeline is the same pattern Carnot needs. The 100% accuracy / 0% FP claim directly addresses the ArithmeticExtractor's false positive problem. Key technique: use LLM to generate Z3 code from chain-of-thought, then run Z3 to check satisfiability.
- **When to pursue:** Z3 extraction milestone. Use as implementation blueprint for NL→Z3 auto-spec.

### Correctness-Guaranteed Code Generation via Constrained Decoding
- **Paper:** arxiv.org/abs/2508.15866 (2025-08)
- **What:** Constrained decoding algorithm using a context-sensitive parser that outputs regular expressions constraining each generation step. Dynamic tree of parsers with variable scopes and type constraints. Guarantees structural correctness of generated code.
- **Relevance:** Most direct published work for Carnot's guided decoding capability (FR-12). The context-sensitive parser approach is more principled than Carnot's current logit-adjustment approach. The dynamic constraint tree maps onto Carnot's ConstraintStateMachine (Exp 125). Applicable to HumanEval code generation verification.
- **When to pursue:** Guided decoding improvement milestone. Borrow the parser-tree constraint architecture.

### Hybrid FPGA Ising Decomposition with COBI Chip
- **Paper:** arxiv.org/abs/2602.15985 (2026-02)
- **What:** Hybrid FPGA-based decomposer co-located with COBI Ising chip (50 coupled ring oscillators, 28nm CMOS). Solves sub-problems in 77.5 microseconds with <10mW power. Uses graph decomposition to partition large Ising problems across COBI chips.
- **Relevance:** Directly applicable to KV260 FPGA bring-up (FpgaBackend). The graph decomposition strategy (split large Ising problems into FPGA-solvable sub-problems) can be implemented in software on the KV260 before physical Ising chips are available. The 77.5μs convergence time is the target for KV260 overlay implementation.
- **When to pursue:** KV260 FPGA bring-up. Use decomposition strategy for large constraint graphs.

### A Theoretical Lens for RL-Tuned LLMs via EBMs
- **Paper:** arxiv.org/abs/2512.18730 (2025-12)
- **What:** Analyzes optimal policy in KL-regularized RL; shows natural emergence of EBM form. The optimal RL-tuned LLM IS an EBM over its base model distribution, with the reward as the energy function. Provides theoretical grounding for EBM-based policy learning.
- **Relevance:** Theoretical foundation connecting Carnot's constraint energy to RLHF. If the optimal policy is an EBM, then Carnot's energy-based verification is not just post-hoc checking — it's defining the training objective. Opens a path to RL-based constraint learning (train the LLM to minimize Carnot's energy function). Also validates the EBM-RL connection for the guided decoding adapter.
- **When to pursue:** Long-term guided decoding + self-learning. Informs Tier 4 adaptive structure.

### Security Vulnerability Detection in LLM-Generated Code via Z3
- **Paper:** arxiv.org/abs/2604.05292 (2026-04)
- **What:** Formal verification of 3,500 code artifacts across 7 LLMs using Z3 SMT solver. Mean vulnerability rate: 55.8%. Demonstrates practical Z3 application to LLM output verification at scale. Maps code patterns to security predicates that Z3 can solve.
- **Relevance:** Validates Z3 for code verification at scale (3,500 artifacts). The vulnerability predicate approach is directly applicable to Carnot's code constraint extraction: map HumanEval solutions to Z3 safety predicates (null dereference, buffer overflow, integer overflow) and run Z3. Could replace or complement PBT-based code verification.
- **When to pursue:** Z3 extraction milestone. Use as implementation reference for code constraint Z3 specs.

### Energy Matching — Unifying Flow Matching and EBMs
- **Paper:** arxiv.org/abs/2504.10612 (2025-04)
- **What:** Unifies flow matching with EBMs. Shows that EBMs handle additional priors and partial observations elegantly. Proposes energy matching objective that trains EBMs without MCMC samples. Efficient training without the sampling bottleneck.
- **Relevance:** Training efficiency for Carnot's EBMs. Current CD training requires MCMC samples (slow). Energy matching avoids this by using flow-matching-inspired objectives. Could enable faster iteration on constraint EBM training (Tier 2 memory consolidation, Tier 3 JEPA training).
- **When to pursue:** EBM training efficiency milestone. Evaluate energy matching vs CD for constraint model training.

### Online Learnability of CoT Verifiers: Soundness/Completeness Trade-offs
- **Paper:** arxiv.org/abs/2603.03538 (2026-03)
- **What:** Formal analysis of what CoT verifiers can and cannot learn online. Characterizes soundness/completeness trade-off as a function of verifier expressivity. Proves bounds on which constraint types are online-learnable and which require offline training data.
- **Relevance:** Provides theoretical bounds for Carnot's verify-repair loop. The soundness/completeness framing maps directly onto Carnot's coverage guarantees. Informs how aggressive the repair cycle should be — over-aggressive repair can break sound correct answers. Directly relevant to the constraint addition approach (Tier 2 → Tier 1).
- **When to pursue:** Self-learning milestone. Use bounds to design constraint addition with soundness guarantees.

### Likelihood-Based Reward Designs for LLM Reasoning (EBM Reward Signal)
- **Paper:** arxiv.org/abs/2602.03979 (2026-02)
- **What:** Log-probability of reference answer as RL reward outperforms binary verifier rewards in both verifiable and non-verifiable settings. Bridges CoT fine-tuning across domains. The log-prob reward is more informative than binary correct/incorrect.
- **Relevance:** Carnot's EBM energy score is itself a log-probability proxy. This paper validates using energy/log-prob as the training signal for Carnot's online learning component (Tier 1), without requiring hard binary labels. Could improve the self-learning tracker by using continuous energy signals instead of binary violation flags.
- **When to pursue:** Self-learning Tier 1 improvement. Replace binary violation signal with continuous log-prob energy signal.

### Hardware Acceleration of Frustrated Lattice Systems via Convolutional RBM
- **Paper:** arxiv.org/abs/2511.20911 (2025-11)
- **What:** FPGA implementation of convolutional Restricted Boltzmann Machine achieves 3–5 orders of magnitude speedup over GPU sampling for Shastry-Sutherland frustrated lattice thermodynamics. Achieves ~10^4x speedup at moderate problem sizes.
- **Relevance:** Most concrete FPGA speedup benchmark directly applicable to Carnot's FPGA Ising tier. Convolutional RBM on FPGA is architecturally close to Carnot's Boltzmann crate. The 10^3–10^5x speedup figure is a concrete planning target for the KV260 hardware path. The convolution structure also applies to Carnot's sparse Ising coupling (Exp 61).
- **When to pursue:** FPGA KV260 milestone. Use convolutional RBM architecture as a target benchmark.

### Semantic Energy — Hallucination Beyond Entropy
- **Paper:** arxiv.org/abs/2508.14496 (2025-08)
- **What:** Combines semantic clustering with a Boltzmann-inspired energy distribution operating
  directly on logits to detect hallucinations where semantic entropy alone fails. Works where
  output logit uncertainty is low but factual content is wrong.
- **Relevance:** Boltzmann energy formulation maps directly onto Carnot's `carnot-boltzmann` tier.
  Complementary signal to Spilled Energy (2602.18671): spilled energy fires on high-entropy
  outputs; semantic energy fires on confident-but-wrong outputs.
- **When to pursue:** Implement alongside SpilledEnergyExtractor. May catch different error classes.

### EBM-CoT — Energy-Based Calibration for Chain-of-Thought
- **Paper:** arxiv.org/abs/2511.07124 (2025-11)
- **What:** Refines latent thought representations through an EBM to improve consistency and
  efficiency of implicit chain-of-thought reasoning. Energy function calibrates which reasoning
  paths are consistent with the initial premise.
- **Relevance:** Direct use-case for Carnot's Gibbs/Boltzmann tiers as a calibration layer on
  top of LLM reasoning chains. Informs the self-learning loop and the JEPA predictor design.
- **When to pursue:** After JEPA training (Tier 3) is proven. Could improve predictor training.

### Solver-Aided Policy Verification for Tool-Augmented LLMs
- **Paper:** arxiv.org/abs/2603.20449 (2026-03)
- **What:** Translates natural-language policies into SMT-LIB-2.0 constraints and checks planned
  tool calls against them using Z3. Natural-language-to-SMT pipeline with 100%+ efficiency gain.
- **Relevance:** Exactly addresses Carnot's constraint extraction bottleneck — the NL→SMT
  translation pipeline is the missing piece between EBM scoring and hard logical constraints.
  More principled than NSVIF (2601.17789) for tool-call verification specifically.
- **When to pursue:** Next milestone. Inform the Z3/SMT extraction experiments.

### VERGE — Z3-Based Formal Refinement for LLM Reasoning
- **Paper:** arxiv.org/abs/2601.20055 (2026-01)
- **What:** Uses Z3 to verify and iteratively refine logical reasoning in LLM outputs. Feedback
  loop between symbolic verification and neural generation. First architecture to prove Z3 can
  serve as the correction oracle in an LLM reasoning pipeline.
- **Relevance:** The feedback loop maps to Carnot's self-learning loop — EBM scores could
  replace or augment Z3 for soft/probabilistic constraints. Directly informs Z3-based extraction.
- **When to pursue:** Z3 extraction milestone. Use VERGE's feedback loop as a model.

### Probabilistic Neuro-Symbolic Layer for Algebraic Constraint Satisfaction
- **Paper:** arxiv.org/abs/2503.19466 (2025-03)
- **What:** Differentiable probabilistic layer guaranteeing satisfaction of non-convex algebraic
  constraints, pluggable into any neural architecture with maximum likelihood training.
- **Relevance:** Could serve as a hard-constraint satisfaction layer on top of Carnot's EBM tiers.
  Enforces algebraic constraints while preserving differentiability for the self-learning loop.
  More general than Πnet (2508.10480).
- **When to pursue:** After continuous relaxation (Exp 64) is extended; plug into repair pathway.

### Set-Valued Prediction with Conformal Coverage for LLMs
- **Paper:** arxiv.org/abs/2603.22966 (2026-03)
- **What:** Risk-controlled set-valued prediction using split conformal prediction for LLMs.
  Establishes statistical reliability guarantees on verification scores without retraining.
- **Relevance:** Provides calibrated verification thresholds — conformal coverage bounds give
  statistically valid confidence intervals on EBM-based verification scores. Directly applicable
  to calibrating the PredictiveVerifier gate (Tier 3).
- **When to pursue:** Calibrated verification milestone. Complement JEPA training with conformal bounds.

### Denoising Thermodynamic Models for Probabilistic Hardware
- **Paper:** arxiv.org/abs/2510.23972 (2025-10)
- **What:** First scalable method for applying probabilistic hardware to ML. Runs Denoising
  Thermodynamic Models (DTMs) on probabilistic hardware instead of monolithic EBMs.
  More hardware-efficient than raw EBMs for sampling acceleration.
- **Relevance:** Most directly relevant to Carnot's FPGA/TSU hardware tier. The DTM architecture
  may be more hardware-efficient than raw Ising for the FPGA sampling acceleration goal.
  Connect to the KV260 FpgaBackend design.
- **When to pursue:** FPGA hardware milestone. Consider DTM architecture alongside Ising for KV260.

### KANELÉ — KANs on FPGAs via LUT Evaluation
- **Paper:** arxiv.org/abs/2512.12850 (2025-12)
- **What:** KANs evaluated via look-up tables (LUTs) on FPGAs. Orders-of-magnitude more
  hardware-efficient than floating-point KAN evaluation. LUT-based spline evaluation maps
  naturally to FPGA synthesis.
- **Relevance:** Critical for `carnot-kan` hardware deployment. KAN energy tier + FPGA =
  hardware-accelerated nonlinear constraint verification without full reconfiguration.
  Hardware path for Tier 4 adaptive structure (KAN splines as reprogrammable LUTs).
- **When to pursue:** After KV260 FPGA baseline is proven. Add KAN LUT evaluation to FpgaBackend.

### Generative Thermodynamic Computing
- **Paper:** arxiv.org/abs/2506.15121 (2025-06)
- **What:** Generative modeling framework for thermodynamic hardware. Structured data synthesized
  by Langevin dynamics time-evolution of a physical system. Connects to hardware Ising sampling.
- **Relevance:** Connects Carnot's Langevin-based repair to the physical hardware path. The
  framework bridges software Langevin dynamics (Exp 64) and hardware thermodynamic sampling (TSU).
- **When to pursue:** Hardware tier milestone. Conceptual bridge between Carnot repair and TSU.

## ArXiv Scan — Exp 139 (2026-04-11)

Queries: ebm_verification, ising_language, constraint_neural, kan_energy, guided_decoding, fpga_ising, continual_constraint, thermodynamic_sampling  
Total unique papers scanned: 14  
Top 10 selected by relevance score.

### Interpretation of Crystal Energy Landscapes with Kolmogorov-Arnold Networks
- **ArXiv:** [2604.04636](https://arxiv.org/abs/2604.04636)  (2026-04-06)
- **Authors:** Gen Zu, Ning Mao, Claudia Felser et al.
- **Summary:** Characterizing crystalline energy landscapes is essential to predicting thermodynamic stability, electronic structure, and functional behavior. While machine learning (ML) enables rapid property predictions, the "black-box" nature of most models limits their utility for generating new scientific insights. Here, we introduce Kolmogorov-Arnold Networks (KANs) as an interpretable framework to bridge this gap. Unlike conventional neural networks with fixed activation functions, KANs employ learnable functions that reveal underlying physical relationships. We developed the Element-Weighted KAN, a c...
- **Relevance to Carnot:** KAN energy tier (carnot-kan, Exp 108-109) is already implemented. New results on KAN expressiveness or spline approximation quality could guide hyperparameter tuning or motivate a deeper KAN variant.
- **Proposed experiment:** Exp 141 candidate: Apply the paper's spline-depth or basis-function findings to carnot-kan and re-run Exp 109 AUROC benchmark.

### Kolmogorov-Arnold Energy Models: Fast, Interpretable Generative Modeling
- **ArXiv:** [2506.14167](https://arxiv.org/abs/2506.14167)  (2025-06-17)
- **Authors:** Prithvi Raj
- **Summary:** Generative models typically rely on either simple latent priors (e.g., Variational Autoencoders, VAEs), which are efficient but limited, or highly expressive iterative samplers (e.g., Diffusion and Energy-based Models), which are costly and opaque. We introduce the Kolmogorov-Arnold Energy Model (KAEM) to bridge this trade-off and provide a new avenue for latent-space interpretability. Based on a novel interpretation of the Kolmogorov-Arnold Representation Theorem, KAEM imposes a univariate latent structure that enables fast and exact inference via the inverse transform method. With a low-dime...
- **Relevance to Carnot:** KAN energy tier (carnot-kan, Exp 108-109) is already implemented. New results on KAN expressiveness or spline approximation quality could guide hyperparameter tuning or motivate a deeper KAN variant.
- **Proposed experiment:** Exp 141 candidate: Apply the paper's spline-depth or basis-function findings to carnot-kan and re-run Exp 109 AUROC benchmark.

### Opening the Black-Box: Symbolic Regression with Kolmogorov-Arnold Networks for Energy Applications
- **ArXiv:** [2504.03913](https://arxiv.org/abs/2504.03913)  (2025-04-04)
- **Authors:** Nataly R. Panczyk, Omer F. Erdem, Majdi I. Radaideh
- **Summary:** While most modern machine learning methods offer speed and accuracy, few promise interpretability or explainability -- two key features necessary for highly sensitive industries, like medicine, finance, and engineering. Using eight datasets representative of one especially sensitive industry, nuclear power, this work compares a traditional feedforward neural network (FNN) to a Kolmogorov-Arnold Network (KAN). We consider not only model performance and accuracy, but also interpretability through model architecture and explainability through a post-hoc SHAP analysis. In terms of accuracy, we fin...
- **Relevance to Carnot:** KAN energy tier (carnot-kan, Exp 108-109) is already implemented. New results on KAN expressiveness or spline approximation quality could guide hyperparameter tuning or motivate a deeper KAN variant.
- **Proposed experiment:** Exp 141 candidate: Apply the paper's spline-depth or basis-function findings to carnot-kan and re-run Exp 109 AUROC benchmark.

### Decomposing Large-Scale Ising Problems on FPGAs: A Hybrid Hardware Approach
- **ArXiv:** [2602.15985](https://arxiv.org/abs/2602.15985)  (2026-02-17)
- **Authors:** Ruihong Yin, Yue Zheng, Chaohui Li et al.
- **Summary:** Emerging analog computing substrates, such as oscillator-based Ising machines, offer rapid convergence times for combinatorial optimization but often suffer from limited scalability due to physical implementation constraints. To tackle real-world problems involving thousands of variables, problem decomposition is required; however, performing this step on standard CPUs introduces significant latency, preventing the high-speed solver from operating at full capacity. This work presents a heterogeneous system that offloads the decomposition workload to an FPGA, tightly integrated with a custom 28...
- **Relevance to Carnot:** Direct hardware path for Carnot's TSU-simulation backend (research-references.md §FPGA Ising Machine).  Architectural details (bit-width, LFSR design, AXI interface) could accelerate the FpgaBackend prototype for SamplerBackend (Exp 71).
- **Proposed experiment:** Exp 142 candidate: Implement a minimal Verilog Ising cell based on the paper's design, simulate in Verilator, and compare sample quality to CPU ParallelIsingSampler on a 100-variable SAT.

### LoRA-Based Continual Learning with Constraints on Critical Parameter Changes
- **ArXiv:** [2504.13407](https://arxiv.org/abs/2504.13407)  (2025-04-18)
- **Authors:** Shimou Ling, Liang Zhang, Jiangwei Zhao et al.
- **Summary:** LoRA-based continual learning represents a promising avenue for leveraging pre-trained models in downstream continual learning tasks. Recent studies have shown that orthogonal LoRA tuning effectively mitigates forgetting. However, this work unveils that under orthogonal LoRA tuning, the critical parameters for pre-tasks still change notably after learning post-tasks. To address this problem, we directly propose freezing the most critical parameter matrices in the Vision Transformer (ViT) for pre-tasks before learning post-tasks. In addition, building on orthogonal LoRA tuning, we propose ortho...
- **Relevance to Carnot:** Multi-turn agentic verification (Goal #2) requires the constraint model to accumulate knowledge across steps without catastrophic forgetting.  Directly applicable to the LNN-based constraint adaptation explored in Exp 116.
- **Proposed experiment:** Exp 143 candidate: Apply the paper's continual-learning strategy to carnot-gibbs constraint updates across a 5-step reasoning chain and measure constraint retention vs Exp 116 baseline.

### Lagrange Oscillatory Neural Networks for Constraint Satisfaction and Optimization
- **ArXiv:** [2505.07179](https://arxiv.org/abs/2505.07179)  (2025-05-12)
- **Authors:** Corentin Delacour, Bram Haverkort, Filip Sabo et al.
- **Summary:** Physics-inspired computing paradigms are receiving renewed attention to enhance efficiency in compute-intensive tasks such as artificial intelligence and optimization. Similar to Hopfield neural networks, oscillatory neural networks (ONNs) minimize an Ising energy function that embeds the solutions of hard combinatorial optimization problems. Despite their success in solving unconstrained optimization problems, Ising machines still face challenges with constrained problems as they can become trapped in infeasible local minima. In this paper, we introduce a Lagrange ONN (LagONN) designed to esc...
- **Relevance to Carnot:** Constraint reasoning paper relevant to Carnot's constraint extraction and satisfaction pipeline.  Review for novel constraint types or evaluation benchmarks.

### Joint Continual Learning of Local Language Models and Cloud Offloading Decisions with Budget Constraints
- **ArXiv:** [2602.00166](https://arxiv.org/abs/2602.00166)  (2026-01-29)
- **Authors:** Evan Chen, Wenzhi Fang, Shiqiang Wang et al.
- **Summary:** Locally deployed Small Language Models (SLMs) must continually support diverse tasks under strict memory and computation constraints, making selective reliance on cloud Large Language Models (LLMs) unavoidable. Regulating cloud assistance during continual learning is challenging, as naive reward-based reinforcement learning often yields unstable offloading behavior and exacerbates catastrophic forgetting as task distributions shift. We propose DA-GRPO, a dual-advantage extension of Group Relative Policy Optimization that incorporates cloud-usage constraints directly into advantage computation,...
- **Relevance to Carnot:** Multi-turn agentic verification (Goal #2) requires the constraint model to accumulate knowledge across steps without catastrophic forgetting.  Directly applicable to the LNN-based constraint adaptation explored in Exp 116.
- **Proposed experiment:** Exp 143 candidate: Apply the paper's continual-learning strategy to carnot-gibbs constraint updates across a 5-step reasoning chain and measure constraint retention vs Exp 116 baseline.

### Energy-Dissipative Evolutionary Kolmogorov-Arnold Networks for Complex PDE Systems
- **ArXiv:** [2503.01618](https://arxiv.org/abs/2503.01618)  (2025-03-03)
- **Authors:** Guang Lin, Changhong Mou, Jiahao Zhang
- **Summary:** We introduce evolutionary Kolmogorov-Arnold Networks (EvoKAN), a novel framework for solving complex partial differential equations (PDEs). EvoKAN builds on Kolmogorov-Arnold Networks (KANs), where activation functions are spline based and trainable on each edge, offering localized flexibility across multiple scales. Rather than retraining the network repeatedly, EvoKAN encodes only the PDE's initial state during an initial learning phase. The network parameters then evolve numerically, governed by the same PDE, without any additional optimization. By treating these parameters as continuous fu...
- **Relevance to Carnot:** KAN energy tier (carnot-kan, Exp 108-109) is already implemented. New results on KAN expressiveness or spline approximation quality could guide hyperparameter tuning or motivate a deeper KAN variant.
- **Proposed experiment:** Exp 141 candidate: Apply the paper's spline-depth or basis-function findings to carnot-kan and re-run Exp 109 AUROC benchmark.

### T-SKM-Net: Trainable Neural Network Framework for Linear Constraint Satisfaction via Sampling Kaczmarz-Motzkin Method
- **ArXiv:** [2512.10461](https://arxiv.org/abs/2512.10461)  (2025-12-11)
- **Authors:** Haoyu Zhu, Yao Zhang, Jiashen Ren et al.
- **Summary:** Neural network constraint satisfaction is crucial for safety-critical applications such as power system optimization, robotic path planning, and autonomous driving. However, existing constraint satisfaction methods face efficiency-applicability trade-offs, with hard constraint methods suffering from either high computational complexity or restrictive assumptions on constraint structures. The Sampling Kaczmarz-Motzkin (SKM) method is a randomized iterative algorithm for solving large-scale linear inequality systems with favorable convergence properties, but its argmax operations introduce non-d...
- **Relevance to Carnot:** Constraint reasoning paper relevant to Carnot's constraint extraction and satisfaction pipeline.  Review for novel constraint types or evaluation benchmarks.

### Ferret: An Efficient Online Continual Learning Framework under Varying Memory Constraints
- **ArXiv:** [2503.12053](https://arxiv.org/abs/2503.12053)  (2025-03-15)
- **Authors:** Yuhao Zhou, Yuxin Tian, Jindi Lv et al.
- **Summary:** In the realm of high-frequency data streams, achieving real-time learning within varying memory constraints is paramount. This paper presents Ferret, a comprehensive framework designed to enhance online accuracy of Online Continual Learning (OCL) algorithms while dynamically adapting to varying memory budgets. Ferret employs a fine-grained pipeline parallelism strategy combined with an iterative gradient compensation algorithm, ensuring seamless handling of high-frequency data with minimal latency, and effectively counteracting the challenge of stale gradients in parallel training. To adapt to...
- **Relevance to Carnot:** Multi-turn agentic verification (Goal #2) requires the constraint model to accumulate knowledge across steps without catastrophic forgetting.  Directly applicable to the LNN-based constraint adaptation explored in Exp 116.
- **Proposed experiment:** Exp 143 candidate: Apply the paper's continual-learning strategy to carnot-gibbs constraint updates across a 5-step reasoning chain and measure constraint retention vs Exp 116 baseline.

### Proposed Experiments for Milestone 2026.04.10

#### EXP-140: Constraint-Projection Guided Decoding Latency Benchmark
- **Goal:** Goal #4 — Guided decoding latency benchmark
- **Spec:** REQ-GUIDED-001, SCENARIO-GUIDED-002
- **Complexity:** medium
- **Description:** Implement a per-token constraint-projection operator in the EnergyGuidedSampler that projects logits onto a constraint-satisfying subspace using the KAN energy gradient.  Measure wall-clock overhead per token at batch sizes 1, 8, 32 on CPU.  Success criterion: <1 ms per token at batch=1 (Exp 102 budget).  Compare to Exp 138's alpha-penalty approach.  This directly addresses Goal #4 (guided decoding latency) and produces publishable numbers for the HuggingFace model card.

#### EXP-141: Apple GSM8K Adversarial Benchmark — Carnot vs LLM Baseline
- **Goal:** Goal #5 — Apple GSM8K adversarial benchmark
- **Spec:** REQ-VERIFY-002, SCENARIO-VERIFY-005
- **Complexity:** medium
- **Description:** Run Carnot's verify-repair pipeline on the Apple GSM8K adversarial variant (arxiv 2410.05229): same problems with swapped numbers and one irrelevant sentence added.  Measure: (a) LLM accuracy drop on adversarial vs standard, (b) Carnot accuracy on adversarial, (c) delta between Carnot improvement on adversarial vs standard.  Expected result: improvement is larger on adversarial because there are more arithmetic errors to catch via Ising constraint checking.  This is the single most credibility-building experiment available and directly tests the core thesis.

#### EXP-142: Multi-Turn Constraint Propagation — 3-Step Reasoning Chain
- **Goal:** Goal #2 — Multi-turn agentic verification
- **Spec:** REQ-MULTITURN-001, SCENARIO-MULTITURN-001
- **Complexity:** high
- **Description:** Extend the verify-repair loop (Exp 57) to a 3-step chain: plan → calculate → conclude.  Each step's verified facts become hard constraints on the next step.  Measure constraint retention rate (what fraction of step-1 constraints are still satisfied at step 3) and overall accuracy on a 50-problem multi-step arithmetic dataset.  Directly addresses Goal #2 (multi-turn agentic verification) and produces the first multi-step constraint propagation numbers for the project.


## ArXiv Scan — Exp 165 (2026-04-11, Planning for Milestone 2026.04.12)

Queries: ebm_constraint_verification, jepa_partial_prediction, kan_energy_2026,
ising_constraint_neural, spilled_energy_llm, guided_decoding_energy, continual_constraint_learning,
thermodynamic_fpga_2025, factual_verification_kb, autoregressive_ebm

### Autoregressive Language Models are Secretly Energy-Based Models
- **ArXiv:** [2512.15605](https://arxiv.org/abs/2512.15605) (2025-12)
- **What:** Establishes an explicit bijection between autoregressive LLMs and EBMs in function
  space. The key insight: next-token prediction naturally computes a "lookahead energy" — the
  negative log-probability of a continuation under the model. This energy is computable without
  fine-tuning. Shows that LLMs implicitly optimize an energy objective during inference.
- **Relevance to Carnot:** Theoretical foundation for using LLM logits directly as EBM energy
  signals without external KB. The "lookahead energy" from 2512.15605 is complementary to
  "spilled energy" (2602.18671): spilled energy measures token-level uncertainty; lookahead
  energy measures continuation-level constraint coherence. Together they form a richer signal.
  Could enable Carnot to extract constraint-quality estimates directly from the LLM's own
  token predictions, bypassing the need for external Ising verification on easy queries.
- **Proposed experiment for Exp 169:** Implement LookaheadEnergyExtractor using the AR-EBM
  bijection. Measure: does lookahead energy predict constraint violations better than spilled
  energy alone? Run on same TruthfulQA and GSM8K samples. If AUROC > 0.70 → add to pipeline
  as a fast pre-filter before Ising.

### Thermodynamic Computing System for AI Applications
- **Paper:** [s41467-025-59011-x](https://www.nature.com/articles/s41467-025-59011-x)
  (Nature Communications, 2025)
- **What:** Implements a stochastic processing unit (SPU) using RLC circuits controlled by
  FPGA. The circuits naturally sample from Boltzmann distributions — thermodynamic noise IS
  the computation. Used for MCMC sampling and linear algebra. Demonstrates 100x energy
  efficiency vs GPU for Boltzmann sampling.
- **Relevance to Carnot:** Validates the hardware path for thermodynamic/Ising sampling.
  The FPGA-RLC hybrid is a near-term implementation of what Extropic's Z1 will do at larger
  scale. Carnot's SamplerBackend (Exp 71) is designed exactly for this: swap CpuBackend for
  FpgaBackend (or SPU backend) without changing the pipeline. The paper's FPGA control logic
  would map directly to a Kria KV260 + analog frontend circuit.
- **When to pursue:** When FPGA hardware is available. Read this paper for the FPGA controller
  design before implementing FpgaBackend.

### Foundations of Global Consistency Checking with Noisy LLM Oracles
- **ArXiv:** [2601.13600](https://arxiv.org/abs/2601.13600) (2026-01)
- **What:** Addresses the problem of verifying global consistency of LLM outputs when the
  verifier itself (the LLM) is noisy. Introduces structured checking frameworks — checking
  multiple outputs for pairwise consistency using a noisy oracle. Shows that majority-voting
  consistency checks converge even with 30% oracle error rate.
- **Relevance to Carnot:** Directly applicable to multi-turn agentic verification (Goal #2).
  When VerifyRepairPipeline acts as the "oracle" across a reasoning chain, it may make errors.
  The consistency checking framework from 2601.13600 provides the theoretical basis for
  ConstraintStateMachine's contradiction detection: if step 3's verified output contradicts
  step 1's verified fact, a global inconsistency exists even if each step passed locally.
- **Proposed experiment for Exp 176:** Implement GlobalConsistencyChecker using 2601.13600's
  pairwise consistency framework. Test on 20 multi-step factual reasoning chains. Measure:
  fraction of globally inconsistent chains detected that local step-by-step checking misses.

### VALENCE-SALS — RT-Core Geometric Constraint Indexing
- **Repo:** github.com/PaperScarecrow/VALENCE-SALS
- **Paper:** zenodo.org/records/19421339
- **What:** Replaces O(N²) transformer attention with O(log N) GPU ray-tracing
  through a 3D BVH of UMAP-projected word embeddings. Vulkan RT cores do
  spatial search instead of dot-product attention. ~1.2GB VRAM, 20-30W.
- **Relevance to Carnot:**
  1. **Geometric constraint indexing** — Map constraint types into 3D space
     where related constraints cluster. Fire rays to find relevant constraints
     for a given input in O(log N) instead of running all 5+ extractors.
     Matters when we have 100+ constraint types.
  2. **RT-core Ising coupling lookup** — BVH over sparse coupling graph could
     skip zero-couplings in O(log N). Only relevant at 50K+ vars where O(N²)
     dominates. FPGA is a more direct path for us.
- **Limitation:** VALENCE does retrieval only, not generation or verification.
  The "thermodynamic" framing is metaphorical, not actual energy minimization.
- **When to pursue:** When constraint type count exceeds ~50 and AutoExtractor
  becomes a bottleneck. Low priority currently.

## ArXiv Scan — Exp 165 (20260411)

Queries: jepa_prediction, factual_verification_kg, fpga_ising_2026, ebm_reasoning_verification, orthogonal_projection, ebm_hallucination, continual_forgetting, kan_interpretable, guided_decoding_2026, thermodynamic_hardware  
Total unique new papers scanned: 29  
Deduplicated against Exp 139: 0 papers skipped.  
Top 10 selected by relevance score.

### Cram Less to Fit More: Training Data Pruning Improves Memorization of Facts
- **ArXiv:** [2604.08519](https://arxiv.org/abs/2604.08519)  (2026-04-09)
- **Authors:** Jiayuan Ye, Vitaly Feldman, Kunal Talwar
- **Summary:** Large language models (LLMs) can struggle to memorize factual knowledge in their parameters, often leading to hallucinations and poor performance on knowledge-intensive tasks. In this paper, we formalize fact memorization from an information-theoretic perspective and study how training data distributions affect fact accuracy. We show that fact accuracy is suboptimal (below the capacity limit) whenever the amount of information contained in the training data facts exceeds model capacity. This is further exacerbated when the fact frequency distribution is skewed (e.g. a power law). We propose da...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Ads in AI Chatbots? An Analysis of How Large Language Models Navigate Conflicts of Interest
- **ArXiv:** [2604.08525](https://arxiv.org/abs/2604.08525)  (2026-04-09)
- **Authors:** Addison J. Wu, Ryan Liu, Shuyue Stella Li et al.
- **Summary:** Today's large language models (LLMs) are trained to align with user preferences through methods such as reinforcement learning. Yet models are beginning to be deployed not merely to satisfy users, but also to generate revenue for the companies that created them through advertisements. This creates the potential for LLMs to face conflicts of interest, where the most beneficial response to a user may not be aligned with the company's incentives. For instance, a sponsored product may be more expensive but otherwise equal to another; in this case, what does (and should) the LLM recommend to the us...
- **Relevance to Carnot:** Ising-language model intersection maps directly to Carnot's constraint extraction → Ising energy pipeline. New coupling structures or sampling tricks could improve Exp 55/62/88 results.
- **Proposed experiment:** Exp 167 candidate: Replace Carnot's current Ising coupling initialisation with the paper's method and compare constraint satisfaction rate on the GSM8K adversarial benchmark.

### Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction
- **ArXiv:** [2604.08542](https://arxiv.org/abs/2604.08542)  (2026-04-09)
- **Authors:** Tao Xie, Peishan Yang, Yudong Jin et al.
- **Summary:** This paper addresses the task of large-scale 3D scene reconstruction from long video sequences. Recent feed-forward reconstruction models have shown promising results by directly regressing 3D geometry from RGB images without explicit 3D priors or geometric constraints. However, these methods often struggle to maintain reconstruction accuracy and consistency over long sequences due to limited memory capacity and the inability to effectively capture global contextual cues. In contrast, humans can naturally exploit the global understanding of the scene to inform local perception. Motivated by th...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models
- **ArXiv:** [2604.08527](https://arxiv.org/abs/2604.08527)  (2026-04-09)
- **Authors:** Feng Luo, Yu-Neng Chuang, Guanchu Wang et al.
- **Summary:** On-policy distillation (OPD) trains student models under their own induced distribution while leveraging supervision from stronger teachers. We identify a failure mode of OPD: as training progresses, on-policy rollouts can undergo abrupt length inflation, causing truncated trajectories to dominate the training data. This truncation collapse coincides with abrupt repetition saturation and induces biased gradient signals, leading to severe training instability and sharp degradation in validation performance. We attribute this problem to the interaction between student-induced data collection and...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Johnson-Schwartzman Gap Labelling for Metric and Discrete Decorated Graphs
- **ArXiv:** [2604.08496](https://arxiv.org/abs/2604.08496)  (2026-04-09)
- **Authors:** Ram Band, Gilad Sofer
- **Summary:** We study Schrödinger operators on metric and discrete decorated graphs. The values taken by the integrated density of states (IDS) on spectral gaps are called gap labels. A natural question is which gap labels can occur. We answer this for graphs arising from uniquely ergodic one-dimensional dynamical systems by proving Johnson-Schwartzman gap-labelling theorems in both the metric and discrete settings. Our results extend Johnson-Schwartzman gap labelling beyond the standard one-dimensional setting. Unlike in one dimension, these graphs may contain cycles, which prevent the use of Sturm oscill...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Wideband Compressed-Domain Cramér--Rao Bounds for Near-Field XL-MIMO: Data and Geometric Diversity Decomposition
- **ArXiv:** [2604.08531](https://arxiv.org/abs/2604.08531)  (2026-04-09)
- **Authors:** Rıfat Volkan Şenyuva
- **Summary:** Wideband orthogonal frequency-division multiplexing (OFDM) over extremely large-scale MIMO (XL-MIMO) arrays in the near-field Fresnel regime suffers from a coupled beam-squint and wavefront-curvature effect that renders single-frequency covariance models severely biased: the per-subcarrier compressed covariance diverges from the center-frequency model by 64\% at $B = 100$~MHz and by 177\% at $B = 400$~MHz. We derive the wideband compressed-domain Cramér--Rao bound (CRB) for hybrid analog--digital architectures and decompose the Fisher information gain into a dominant data-diversity term that s...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### FIT: A Large-Scale Dataset for Fit-Aware Virtual Try-On
- **ArXiv:** [2604.08526](https://arxiv.org/abs/2604.08526)  (2026-04-09)
- **Authors:** Johanna Karras, Yuanhao Wang, Yingwei Li et al.
- **Summary:** Given a person and a garment image, virtual try-on (VTO) aims to synthesize a realistic image of the person wearing the garment, while preserving their original pose and identity. Although recent VTO methods excel at visualizing garment appearance, they largely overlook a crucial aspect of the try-on experience: the accuracy of garment fit -- for example, depicting how an extra-large shirt looks on an extra-small person. A key obstacle is the absence of datasets that provide precise garment and body size information, particularly for "ill-fit" cases, where garments are significantly too large...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Disentangling cosmic distance tensions with early and late dark energy
- **ArXiv:** [2604.08530](https://arxiv.org/abs/2604.08530)  (2026-04-09)
- **Authors:** Tanisha Jhaveri, Tanvi Karwal, Thomas Crawford et al.
- **Summary:** Recent cosmological data reveal tension between parameters inferred from measurements of the cosmic microwave background (CMB), baryon acoustic oscillations (BAO), and supernovae (SN) under $Λ$CDM. Typical dynamical dark energy parameterizations (such as $w_0w_a$) that seek to jointly resolve these tensions have an equation of state parameter that crosses into the phantom regime, leading to potential instabilities for physical models. We show that the BAO (early-time) and SN (late-time) sides of the tension can instead be treated independently. Early dark energy (EDE) can reduce the tension be...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Measurement-induced state transitions across the fluxonium qubit landscape
- **ArXiv:** [2604.08515](https://arxiv.org/abs/2604.08515)  (2026-04-09)
- **Authors:** Alex A. Chapple, Boris M. Varbanov, Alexander McDonald et al.
- **Summary:** Understanding the mechanisms that limit high-fidelity readout in circuit quantum electrodynamics is essential for its optimization. Multi-photon resonances are understood to be a limiting factor, causing population transfer from the computational states to higher-energy states under drive. This effect, known as measurement-induced state transitions, has been extensively studied for the transmon qubit. While this exploration has begun for the fluxonium qubit, a systematic study of this effect is lacking. Here, we bridge this gap by theoretically studying measurement-induced state transitions in...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### MolmoWeb: Open Visual Web Agent and Open Data for the Open Web
- **ArXiv:** [2604.08516](https://arxiv.org/abs/2604.08516)  (2026-04-09)
- **Authors:** Tanmay Gupta, Piper Wolters, Zixian Ma et al.
- **Summary:** Web agents--autonomous systems that navigate and execute tasks on the web on behalf of users--have the potential to transform how people interact with the digital world. However, the most capable web agents today rely on proprietary models with undisclosed training data and recipes, limiting scientific understanding, reproducibility, and community-driven progress. We believe agents for the open web should be built in the open. To this end, we introduce (1) MolmoWebMix, a large and diverse mixture of browser task demonstrations and web-GUI perception data and (2) MolmoWeb, a family of fully ope...
- **Relevance to Carnot:** General relevance to energy-based models or neural constraint satisfaction. Review for techniques applicable to Carnot's sampling or verification pipeline.

### Proposed Experiments for Milestone 2026.04.12

#### EXP-166: JEPA Fast-Path Violation Predictor
- **Goal:** Goal #4 — Guided decoding latency; JEPA fast-path research direction
- **Spec:** REQ-GUIDED-001, SCENARIO-GUIDED-002, REQ-RESEARCH-001
- **Complexity:** medium
- **Description:** Train a small (≤10 M param) JEPA-style joint-embedding predictor on (partial_response_prefix, constraint_violation_flag) pairs derived from accumulated verify-repair logs (Exps 57/96/138). The predictor reads the first 50% of tokens and predicts whether the completed response will violate any active constraint. If confidence is high, skip the full Ising verification pass (fast-path). If confidence is low or violation likely, trigger full verification. Measure: AUROC of violation prediction, false-negative rate (missed violations), and net latency saving vs always-full-verify. Success: >0.85 AUROC and >40% latency reduction on the Exp 138 benchmark trace, with zero false-negative budget exceeded. Models: Qwen3.5-0.8B and google/gemma-4-E4B-it as LLM backends.

#### EXP-167: Orthogonal Projection Constraint Repair in EnergyGuidedSampler
- **Goal:** Goal #4 — Guided decoding; orthogonal projection repair direction
- **Spec:** REQ-GUIDED-001, SCENARIO-GUIDED-002
- **Complexity:** medium
- **Description:** Replace the current alpha-penalty logit adjustment in EnergyGuidedSampler with an orthogonal projection operator: given the set of active constraints as linear inequalities in logit space, project the model's logit vector onto the feasible polytope using a fast iterative solver (e.g., Dykstra's algorithm). This guarantees hard constraint satisfaction (CSR=100%) without penalty tuning. Benchmark: (a) Constraint satisfaction rate on 100 arithmetic generation tasks, (b) per-token latency at batch=1 on CPU, (c) generation quality (BLEU vs unconstrained). Success: CSR=100% with <1 ms per token and BLEU within 5% of unconstrained. This directly addresses the open question from Exp 138 (alpha-tuning instability) and could become the canonical guided decoding method for Carnot. Models: Qwen3.5-0.8B and google/gemma-4-E4B-it.

#### EXP-168: Knowledge-Graph Factual Constraint Extraction — TriviaQA Pilot
- **Goal:** Goal #3 — Factual extractor; knowledge-graph constraint grounding
- **Spec:** REQ-FACTUAL-001, SCENARIO-FACTUAL-001, REQ-RESEARCH-001
- **Complexity:** high
- **Description:** Extend Carnot's constraint extraction pipeline (currently arithmetic-only) to factual claims grounded in a lightweight knowledge graph. Use Wikidata SPARQL (public endpoint, no cost) to resolve named entities and their relations as soft constraints encoded into Ising couplings. Run on 100 TriviaQA questions using Qwen3.5-0.8B as the LLM. Measure: (a) factual claim detection rate, (b) KG resolution success rate, (c) Ising verifier precision/recall on factual hallucinations vs arithmetic-only baseline. Success: ≥60% factual claim detection with ≥70% verifier precision — establishing the first data point for Goal #3 (factual extractor). Also test with google/gemma-4-E4B-it to check model-agnosticism.

<!-- EXP210_REFERENCES_START -->
## 2026-04-12 - Exp 210: Constraint Extraction for Instruction-Tuned Models

### Core papers
- **#1 Neuro-Symbolic Verification on Instruction Following of LLMs** (arXiv 2601.17789) - https://arxiv.org/abs/2601.17789
  Why it matters: Most direct fit to Carnot's current blocker: it treats instruction following as a constraint-satisfaction problem and combines logical plus semantic checks in one verifier.
  Carnot use: Use as the primary template for a prompt-to-constraint intermediate representation plus solver-backed verification path.
- **#2 ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming** (EMNLP 2025) - https://aclanthology.org/2025.emnlp-main.809/
  Why it matters: Shows that instruction-tuned models can be specialized for constraint programming, paired with retrieval and guided self-correction, and evaluated on an industrial benchmark.
  Carnot use: Borrow the CP modeling pattern for scheduling and resource constraints, especially as a second solver route beside Z3.
- **#3 LLM Self-Correction with DeCRIM: Decompose, Critique, and Refine for Enhanced Following of Instructions with Multiple Constraints** (Findings of EMNLP 2024) - https://aclanthology.org/2024.findings-emnlp.458/
  Why it matters: Directly decomposes instructions into atomic constraints, then critiques failures at the constraint level using RealInstruct and IFEval.
  Carnot use: Build Carnot's first prompt-side atomic constraint extractor and use DeCRIM-style critique labels as supervision.
- **#4 CARE-STaR: Constraint-aware Self-taught Reasoner** (Findings of ACL 2025) - https://aclanthology.org/2025.findings-acl.1116/
  Why it matters: Separates easy versus ambiguous constraints and learns different reasoning traces for different constraint levels.
  Carnot use: Route high-confidence literal constraints to symbolic solvers and keep ambiguous constraints on a softer verification path.
- **#5 VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks** (arXiv 2511.04662) - https://arxiv.org/abs/2511.04662
  Why it matters: Formalizes each chain-of-thought step into first-order logic and checks whether each step is grounded in source context, commonsense, or prior steps.
  Carnot use: Prototype a typed step graph for arithmetic and logic traces instead of relying on raw free-form chain-of-thought text.
- **#6 PCRLLM: Proof-Carrying Reasoning with Large Language Models under Stepwise Logical Constraints** (arXiv 2511.08392) - https://arxiv.org/abs/2511.08392
  Why it matters: Constrains each reasoning step to explicit premises, rules, and conclusions so chain-level validation becomes possible even for black-box models.
  Carnot use: Adopt the premise-rule-conclusion record format for future step-level verifier experiments.
- **#7 Deductive Verification of Chain-of-Thought Reasoning** (NeurIPS 2023) - https://proceedings.neurips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html
  Why it matters: Still the clearest baseline for decomposing verification into small subprocesses with a constrained natural-language reasoning format.
  Carnot use: Use as the baseline to beat for stepwise validation and premise-minimization prompts.
- **#8 Faithful Logical Reasoning via Symbolic Chain-of-Thought** (ACL 2024) - https://arxiv.org/abs/2405.18357
  Why it matters: Translates natural language reasoning into symbolic expressions, then verifies both translation and reasoning with a verifier.
  Carnot use: Good bridge design for a prompt-answer pair where Carnot first converts text to symbolic state and only then verifies.
- **#9 Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning** (Findings of EMNLP 2023) - https://aclanthology.org/2023.findings-emnlp.248/
  Why it matters: Foundational hybrid pattern: translate to symbolic form, run deterministic inference, and use solver errors to refine the formalization.
  Carnot use: Keep as the minimum viable pattern for solver-backed extraction experiments.
- **#10 Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning** (arXiv 2510.01069) - https://arxiv.org/abs/2510.01069
  Why it matters: Provides a formal lens for mapping informal chain-of-thought into typed proof objects and treating well-typedness as a certificate of faithfulness.
  Carnot use: Use as design guidance for a typed intermediate representation rather than free-form regex over reasoning text.

### Benchmark assets
- **FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models** (2024) - https://aclanthology.org/2024.acl-long.257/ - Seed a prompt-side benchmark with explicit content, situation, style, format, and example constraints.
- **CFBench: A Comprehensive Constraints-Following Benchmark for LLMs** (2025) - https://aclanthology.org/2025.acl-long.1581/ - Adds broader real-life constraint taxonomies and requirement-priority scoring.
- **RealInstruct** (2024) - https://aclanthology.org/2024.findings-emnlp.458/ - Real user multi-constraint instructions are a better supervision source than synthetic prompt lists.
- **VIFBench** (2026) - https://arxiv.org/abs/2601.17789 - Instruction-following verifier benchmark with fine-grained labels; closest external evaluation target to Carnot's gap.
- **IndusCP** (2025) - https://aclanthology.org/2025.emnlp-main.809/ - Industrial constraint-programming tasks for scheduling and resource-allocation extraction.
- **P-FOLIO** (2024) - https://arxiv.org/abs/2410.09207 - Human-written stepwise logical proofs for evaluating step-level reasoning extraction.
- **FormalBench** (2025) - https://aclanthology.org/2025.acl-long.1068/ - Program-semantics benchmark where formal specification inference is the task itself.
- **StructFlowBench** (2025) - https://aclanthology.org/2025.findings-acl.486/ - Useful if Carnot extends from single-turn extraction to multi-turn structural constraints.

### Monitorability and CoT risk evidence
- **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation** (2025) - https://arxiv.org/abs/2503.11926 - Shows chain-of-thought can help oversight, but strong optimization pressure can induce obfuscated reasoning.
- **Can Reasoning Models Obfuscate Reasoning? Stress-Testing Chain-of-Thought Monitorability** (2025) - https://arxiv.org/abs/2510.19851 - Explicit stress test showing some models can hide adversarial intent under obfuscation pressure.
- **Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity** (2025) - https://arxiv.org/abs/2510.27378 - Useful metric design: faithful traces can still be poor monitors when they omit crucial factors.
- **Diagnosing Pathological Chain-of-Thought in Reasoning Models** (2026) - https://arxiv.org/abs/2602.13904 - Gives concrete pathology categories and cheap diagnostics for post-hoc rationalization, encoded reasoning, and internalized reasoning.
- **Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Reasoning Models?** (2026) - https://arxiv.org/abs/2603.22582 - Recent cross-model evidence that faithfulness varies sharply by family and hint type, which argues for model-specific monitorability gates.
<!-- EXP210_REFERENCES_END -->

## 2026-04-12 - Milestone 2026.04.15 Planning Refresh

### Prompt-Side Constraint IR and Constrained Generation
- **ConstraintBench: Benchmarking LLM Constraint Reasoning on Direct Optimization** (arXiv 2602.22465) - https://arxiv.org/abs/2602.22465
  Why it matters: benchmark centered on hard, soft, and compositional constraints instead of generic instruction-following scores.
  Carnot use: seed Exp 211 and Exp 221 with a prompt-to-constraint IR benchmark that measures constraint extraction coverage before repair.
- **CRANE: Reasoning with constrained LLM generation** (ICML 2025 / arXiv 2502.09061) - https://arxiv.org/abs/2502.09061
  Why it matters: shows why tight grammars can suppress reasoning, then fixes that with reasoning-augmented constrained decoding.
  Carnot use: use a typed reasoning grammar that preserves solve quality while forcing monitorable structure for Qwen3.5-0.8B and Gemma4-E4B-it.
- **SynCode: LLM Generation with Grammar Augmentation** (TMLR 2025 / OpenReview) - https://openreview.net/forum?id=HiUZtgAPoH
  Why it matters: practical CFG-constrained decoding with soundness and completeness guarantees for JSON, Python, SQL, and Go.
  Carnot use: candidate backend for emitting typed step graphs and structured verifier payloads in Exp 212 and Exp 216.
- **BEAVER: An Efficient Deterministic LLM Verifier** (2025 preprint PDF) - https://ggndpsngh.github.io/files/BEAVER.pdf
  Why it matters: deterministic probability bounds for prefix-closed semantic constraints are a better fit than rejection sampling when a verifier needs bounded risk, not just empirical pass rates.
  Carnot use: design inspiration for a bounded semantic verifier that prunes impossible continuations early instead of only inspecting final text.

### Semantic Verification and Hallucination Decomposition
- **MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination** (arXiv 2603.24579) - https://arxiv.org/abs/2603.24579
  Why it matters: decomposes answers into atomic claims and routes them to specialized checkers before recombining the verdict.
  Carnot use: direct template for Exp 215's semantic/question-grounding verifier, where current live GSM8K errors are semantic rather than arithmetic.
- **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation** (arXiv 2503.11926) - https://arxiv.org/abs/2503.11926
  Why it matters: makes the monitorability failure mode explicit: free-form reasoning traces can become less useful exactly when oversight pressure increases.
  Carnot use: justifies Exp 213's monitorability audit and fallback policy before Carnot relies on chain-of-thought as a verifier input.
- **Diagnosing Pathological Chain-of-Thought in Reasoning Models** (arXiv 2602.13904) - https://arxiv.org/abs/2602.13904
  Why it matters: supplies concrete pathology labels for post-hoc rationalization, hidden reasoning, and encoded shortcuts.
  Carnot use: label schema for Exp 214's semantic failure corpus and the structured fallback rules in Exp 213.

### Code Verification and Formalization
- **Use Property-Based Testing to Bridge LLM Code Generation and Validation** (arXiv 2506.18315) - https://arxiv.org/abs/2506.18315
  Why it matters: property synthesis closes the gap between weak prompt-derived tests and stronger verifier feedback for generated code.
  Carnot use: strongest direct follow-on to Exp 208; use property-generated tests and invariants to improve repair on HumanEval without changing the base models.
- **Logical Intelligence / Aleph Prover update** (Jan 6, 2026) - https://logicalintelligence.com/aleph-prover-1000.html
  Why it matters: machine-checked proof generation is now strong enough to be a realistic north star for formal code verification rather than a distant aspirational target.
  Carnot use: not a next-milestone dependency, but a useful end-state reference for where code verification should head after runtime and property-based checks mature.

### Continuous Self-Learning and EBM Follow-Ons
- **Project Aletheia: Verifier-Guided Distillation of Backtracking for Small Language Models** (arXiv 2601.14290) - https://arxiv.org/abs/2601.14290
  Why it matters: learns from verifier-approved backtracking traces instead of static supervised targets.
  Carnot use: direct inspiration for Exp 222 and Exp 223, where live verifier traces become reusable memory and repair policy updates.
- **Semantic Scholar citation sweep for EBT (2507.02092)** - https://www.semanticscholar.org/paper/Energy-Based-Transformers-are-Scalable-Learners-and-Gladstone-Nanduru/2da9163730998a4368c609972ccff0582518b36b
  Why it matters: the most actionable citations are `A Pipeline for Assessing Metacognitive Reasoning in Energy-Based Transformers while Generating Code` and `Transformers as Intrinsic Optimizers: Forward Inference through the Energy Principle`, both of which point toward verifier-in-the-loop reasoning rather than pure architecture replacement.
  Carnot use: reinforces the choice to push Carnot's code-verification and trace-learning loops first; the ARM-EBM citation tree is still too young to drive near-term experiments by itself.

### Hardware Watch
- **Extropic hardware direction (TSU / XTR-0)** - https://extropic.ai/
  Why it matters: Extropic is now explicitly positioning XTR-0 as the bridge between conventional processors and future thermodynamic chips, which makes software-side algorithm design on classical hardware more valuable right now.
  Carnot use: keep the TSU path alive, but do not block 2026.04.15 on new hardware; the next milestone should win on CUDA + CPU first, then hand cleaner verifier workloads to FPGA/TSU prototypes later.

## 2026-04-12 - Milestone 2026.04.17 Planning Refresh

### Calibrated Semantic Verification
- **$V_1$: Unifying Generation and Self-Verification for Parallel Reasoners** (arXiv 2603.04304) - https://arxiv.org/abs/2603.04304
  Why it matters: replaces weak pointwise scoring with pairwise self-verification and uncertainty-guided verification compute allocation.
  Carnot use: build a pairwise verifier that compares baseline vs repaired candidates on GSM8K-style semantic failures and code repairs, instead of trusting scalar pointwise scores.
- **MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination** (arXiv 2603.24579) - https://arxiv.org/abs/2603.24579
  Why it matters: decomposes answers into atomic propositions and validates them with deliberate information asymmetry so the checker cannot simply echo the generator's mistakes.
  Carnot use: implement a claim-isolated checker path for typed reasoning IR where the checker only sees the prompt, extracted claim, and evidence slice.
- **Semantic Energy: Detecting LLM Hallucination Beyond Entropy** (arXiv 2508.14496) - https://arxiv.org/abs/2508.14496
  Why it matters: uses a Boltzmann-style energy over clustered semantics rather than plain entropy, giving a stronger uncertainty signal.
  Carnot use: add a cheap energy prior for semantic verifier calibration and false-positive suppression before full repair is triggered.
- **Weaver: Shrinking the Generation-Verification Gap by Scaling Compute for Verification** (OpenReview, 2025) - https://openreview.net/forum?id=dRjt4vlYVQ
  Why it matters: shows that weakly supervised verifier ensembles and repeated-sampling selection can materially improve reasoning accuracy.
  Carnot use: treat tracker, semantic checker, and code-property verifier as an ensemble instead of a single verdict source.

### Structured Outputs and Constrained Generation
- **Generating Structured Outputs from Language Models: Benchmark and Studies / JSONSchemaBench** (arXiv 2501.10868) - https://arxiv.org/abs/2501.10868
  Why it matters: gives a 10K real-world schema benchmark and explicitly measures efficiency, coverage, and output quality of structured generation systems.
  Carnot use: use JSONSchemaBench-style coverage accounting for typed reasoning / monitorable-output paths instead of only measuring parse success.
- **PSC: Efficient Grammar-Constrained Decoding via Parser Stack Classification** (OpenReview, ICLR 2026 submission) - https://openreview.net/forum?id=SEjxNfQTHN
  Why it matters: computes the full token mask from a single parser-stack classification step, making grammar-constrained decoding much cheaper.
  Carnot use: if structured reasoning remains helpful, PSC-style preprocessing is a plausible path to low-overhead typed emission for Qwen3.5-0.8B and Gemma4-E4B-it.
- **Constrained Decoding of Diffusion LLMs with Context-Free Grammars** (arXiv 2508.10111) - https://arxiv.org/abs/2508.10111
  Why it matters: generalizes constrained decoding to additive infilling and diffusion-style generation while keeping syntax guarantees practical.
  Carnot use: informs future non-autoregressive or infilling-style constrained generation experiments, especially for code repair and structured IR completion.
- **GitHub repo: guidance-ai/jsonschemabench** - https://github.com/guidance-ai/jsonschemabench
  Why it matters: ready-to-run benchmark assets with ~9.6K real-world schemas and MaskBench for mask-computation timing.
  Carnot use: practical external fixture source for measuring monitorable-output engines and schema-feature coverage.

### Code Verification and Formalization
- **Learning to Solve and Verify: A Self-Play Framework for Code and Test Generation** (arXiv 2502.14948) - https://arxiv.org/abs/2502.14948
  Why it matters: solver-verifier self-play materially improves both code and test generation without a larger teacher model.
  Carnot use: direct template for turning Exp 226/227 repair traces into better property generation and better repair prompts.
- **Automatic Generation of Formal Specification and Verification Annotations Using LLMs and Test Oracles** (arXiv 2601.12845) - https://arxiv.org/abs/2601.12845
  Why it matters: verifier feedback plus test-oracle signals produced correct Dafny annotations for 98.2% of 110 programs within 8 iterations.
  Carnot use: motivates a lightweight formal-spec layer for code prompts, where natural-language intent is converted into explicit contracts before repair.
- **Towards Formal Verification of LLM-Generated Code from Natural Language Prompts** (arXiv 2507.13290) - https://arxiv.org/abs/2507.13290
  Why it matters: Astrogator introduces a formal query language plus symbolic verification against user intent, not just execution traces.
  Carnot use: strong design reference for extracting typed code intent from HumanEval prompts and using that intent as a verifier target.

### Continuous Self-Learning and Constraint Systems
- **T-SKM-Net: Trainable Neural Network Framework for Linear Constraint Satisfaction via Sampling Kaczmarz-Motzkin Method** (arXiv 2512.10461) - https://arxiv.org/abs/2512.10461
  Why it matters: integrates randomized SKM-style projection into neural constraint satisfaction with zero violations and practical inference speed.
  Carnot use: candidate repair mechanism for linearized typed constraints and structured-output corrections where Langevin-style repair is too noisy.
- **Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models** (arXiv 2603.12248) - https://arxiv.org/abs/2603.12248
  Why it matters: sequence-level energy-based fine-tuning supplies dense semantic feedback without requiring a task-specific verifier.
  Carnot use: long-term training reference for turning live verifier traces into dense rollout-level objectives instead of sparse accept/reject labels only.

### Semantic Scholar Sweep Around EBT / ARM-EBM
- **Energy-Based Transformers are Scalable Learners and Thinkers** (Semantic Scholar paper page + citations) - https://www.semanticscholar.org/paper/Energy-Based-Transformers-are-Scalable-Learners-and-Gladstone-Nanduru/2da9163730998a4368c609972ccff0582518b36b
  Why it matters: Semantic Scholar already shows concrete follow-on work around EBTs rather than just the original paper.
  Carnot use: the most relevant visible citation targets are:
  1. **Transformers as Intrinsic Optimizers: Forward Inference through the Energy Principle** (arXiv 2511.00907) - https://arxiv.org/abs/2511.00907
     Carnot use: theoretical support for treating transformer forward passes as energy minimization, useful for future guided-decoding or verifier-energy hybrids.
  2. **A Pipeline for Assessing Metacognitive Reasoning in Energy-Based Transformers while Generating Code** (OpenReview, MetaGenAI 2025) - https://openreview.net/forum?id=FrY7CU3U3p
     Carnot use: code-generation-specific EBT evaluation path that may become relevant if Carnot ever hosts a native EBT code verifier instead of only verifier sidecars.
  3. **NRGPT: An Energy-based Alternative for GPT** (arXiv 2512.16762) - https://arxiv.org/abs/2512.16762
     Carnot use: architectural follow-on showing how GPT-like decoding can be reframed as energy-landscape exploration.
- **Autoregressive Language Models are Secretly Energy-Based Models** (arXiv 2512.15605) - https://arxiv.org/abs/2512.15605
  Why it matters: the core ARM-EBM bridge remains strategically important, but the citation graph is still too young to be a reliable roadmap signal.
  Carnot use: keep the paper as theory support for future lookahead-energy experiments, but prioritize experiments with clearer applied leverage this milestone.

### Hardware and Architecture Watch
- **Extropic: Thermodynamic Computing From Zero to One** (Oct 29, 2025) - https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one
  Why it matters: Extropic now has a concrete TSU story, XTR-0 development platform, and a public claim that TSUs compile EBMs down to Gibbs-style local sampling on probabilistic circuits.
  Carnot use: this makes the FPGA/TSU abstraction path more concrete; near-term work should focus on host-side interfaces, sparse couplings, and local-communication-friendly sampling graphs.
- **Extropic hardware page (X0 / XTR-0 / Z1)** - https://extropic.ai/hardware
  Why it matters: XTR-0 is framed as the research platform and Z1 as the production-scale chip with hundreds of thousands of probabilistic circuits per chip and early-access timing in 2026.
  Carnot use: justifies a milestone that finishes KV260 bring-up and host integration now, so Carnot has a cleaner backend boundary before TSU-class hardware is accessible.
- **Logical Intelligence / Kona 1.0** - https://logicalintelligence.com/kona-ebms-energy-based-models
  Why it matters: Kona is now presented as an EBM layer beneath LLMs that enforces validity and safety across system states instead of generating likely text.
  Carnot use: validates the overall product direction, but the immediate takeaway is architectural: keep Carnot focused on verifier-side certainty layers rather than trying to turn the verifier into a chatbot.

## 2026-04-13 - Milestone 2026.04.18 Planning Refresh

### Formal claim verification and solver routing
- **VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning** (arXiv 2601.20055) - https://arxiv.org/abs/2601.20055
  Why it matters: turns natural-language reasoning into typed symbolic claims, routes them through formal solvers, and uses minimal correction sets to localize which assumptions failed.
  Carnot use: strongest direct template for replacing the current claim-isolated semantic verifier with a solver-routed path over typed claims instead of a calibrated scalar-only decision.
- **OpenReview scan outcome: process-verification papers are now explicitly separating final-answer correctness from reasoning integrity** - https://openreview.net/
  Why it matters: the 2026 workshop and submission thread around trustworthy agents reinforces that small models can land on the right answer for the wrong internal process, so process supervision should be treated as a first-class verifier target.
  Carnot use: motivates a reusable process-integrity corpus and additive process verifier for both semantic traces and code-repair traces before trusting apparently correct outputs.

### Code and behavioral verification
- **ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization** (arXiv 2602.15983) - https://arxiv.org/abs/2602.15983
  Why it matters: uses behavioral verification under perturbations to catch solutions that look plausible but fail under slightly altered conditions.
  Carnot use: natural bridge from explicit HumanEval specs to process-aware code verification, especially for detecting "passes one path, breaks nearby paths" repairs on Qwen3.5-0.8B and Gemma4-E4B-it.
- **OpenReview scan outcome: vericoding and formally verified program-synthesis benchmarks are converging on proof-oriented evaluation** - https://openreview.net/
  Why it matters: the benchmark trend is moving toward formal specification fidelity rather than execution success alone.
  Carnot use: useful design pressure for expanding beyond prompt-derived properties into explicitly stated code intent and proof-oriented evaluation slices.

### Constrained generation and structured reasoning
- **OpenReview scan outcome: context-sensitive constrained-decoding work is pushing beyond plain CFG masks** - https://openreview.net/
  Why it matters: newer constrained-decoding work is starting to model obligations that depend on earlier decoded structure rather than only local syntax validity.
  Carnot use: strengthens the case for typed reasoning outputs where later fields depend on earlier commitments, such as claim references, premise bindings, and code-spec clause IDs.

### Hardware-accelerated sampling and sparse connectivity
- **Scalable Connectivity for Ising Machines: From Dense to Sparse** (arXiv 2503.01177) - https://arxiv.org/abs/2503.01177
  Why it matters: shows how dense logical couplings can be compiled into sparse hardware-friendly graphs using copy-node constructions without losing the optimization target.
  Carnot use: direct design input for the next honest FPGA step after Exp 242/243: prepare sparse, partitionable verifier workloads instead of spending milestone slots on more overlay plumbing alone.
- **Decomposing Large-Scale Ising Problems on FPGAs: A Hybrid Hardware Approach** (arXiv 2602.15985) - https://arxiv.org/abs/2602.15985
  Why it matters: argues that host-side decomposition latency can dominate accelerator wins unless the partitioning path is hardware-aware from the start.
  Carnot use: reinforces a milestone design where sparse decomposition and replay-friendly candidate packaging happen in software first, so later KV260 or TSU work accelerates a clean workload rather than an ad hoc one.

### Secondary-source scan outcomes worth tracking
- **Extropic writing / XTR-0 positioning** - https://extropic.ai/writing/
  Why it matters: the current public story emphasizes XTR-0 as a development bridge, not a production replacement for host-side verifier design.
  Carnot use: keep TSU compatibility as a constraint, but prioritize verifier quality and hardware-friendly workload shaping over additional blocked board-bringup tasks next milestone.
- **Logical Intelligence / Kona architecture update** - https://logicalintelligence.com/kona-ebms-energy-based-models
  Why it matters: Kona continues to frame EBMs as a validity layer beneath generators rather than another generator.
  Carnot use: supports a milestone centered on external verification, process integrity, and self-learning sidecars rather than on replacing the target LLMs.

## 2026-04-13 - Milestone 2026.04.19 Planning Refresh

### Predictive Verifier Calibration and Formal Guarantees

- **The 4/δ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantees** (arXiv 2512.02080) - https://arxiv.org/abs/2512.02080
  Why it matters: First formal framework with convergence theorems for multi-stage verification pipelines. Models verification as an absorbing Markov chain with guaranteed termination (the 4/δ bound). Identifies three operating zones — marginal, practical, high-performance — with dynamic calibration strategies and over 90K trial validation.
  Carnot use: Critical for diagnosing the Exp 256 calibration failure where the PredictiveVerifier gate routed everything to FAST_PATH on live data. The Markov chain model directly maps to FormalClaimVerifier → ProcessVerifier → PredictiveVerifier pipeline structure. Provides parameterized calibration strategies for threshold drift handling.

- **TECP: Token-Entropy Conformal Prediction for LLMs** (MDPI Mathematics 2025) - https://www.mdpi.com/2227-7390/13/20/3351
  Why it matters: Uses log-probability token-entropy as a nonconformity score with split conformal prediction, giving finite-sample coverage guarantees on QA benchmarks without live-data retraining. Outperforms self-UQ methods on 6 LLMs.
  Carnot use: Addresses the PredictiveVerifier calibration gap. Token-entropy conformal approach could replace the current static gate threshold with provable constraint satisfaction bounds that adapt to each model's output distribution.

### Self-Play and Verifier Improvement

- **Learning to Rank Chain-of-Thought: Energy Outcome Reward Model (EORM)** (arXiv 2505.14999) - https://arxiv.org/abs/2505.14999
  Why it matters: Lightweight 55M-parameter post-hoc verifier using an energy-based framework for ranking reasoning solutions. Achieves 90.7% on GSM8K with Llama 3 8B, demonstrating practical EBM scaling for solution ranking at inference time.
  Carnot use: Direct parallel to Carnot's PredictiveVerifier. EORM's energy-based ranking at inference time mirrors constraint-energy scoring. Could inform faster ranking strategies for repair candidate selection and provide a published baseline to compare against.

- **Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain** (arXiv 2603.02218) - https://arxiv.org/abs/2603.02218
  Why it matters: Diagnoses why naive self-play fails — the loop synthesizes data without increasing learnable information. Proposes asymmetric co-evolution (weak-to-strong-to-weak), capacity growth matching, and proactive info seeking via external context as remedies.
  Carnot use: Root-cause analysis for the Exp 256 self-learning A/B test failure (constraint templates matched description-text proxies, not real inference tokens). The "proactive info seeking" prescription maps to mining live inference tokens rather than replay description text for constraint template extraction.

- **Propose, Solve, Verify: Self-Play Through Formal Verification (PSV)** (arXiv 2512.18160) - https://arxiv.org/abs/2512.18160
  Why it matters: Uses formal verification (not unit tests) as the self-play training signal. PSV-Verus improves pass@1 by 9.6x over expert-iteration baselines. Formal verification enables robust exploratory data generation, avoiding brittleness of loose test-based supervision.
  Carnot use: Supports upgrading the constraint addition module (Exp 265) to mine formal solver verdicts rather than description-text proximity as the learning signal. Pairs well with the existing FormalClaimVerifier substrate.

### GPU Inference Acceleration

- **Flexible and Efficient Grammar-Constrained Decoding (DOMINO)** (ICML 2025) - https://icml.cc/virtual/2025/poster/43675
  Why it matters: Grammar-constrained decoding with 17.71x faster preprocessing than prior work while maintaining online efficiency. Fully subword-aligned with speculative decoding integration. Enforces CFG rules during generation rather than post-hoc.
  Carnot use: Potential path to encode formal claim templates as grammar constraints, enforcing them during generation rather than in a separate verify step. Speculative decoding compatibility enables pipelining with main generation — relevant for reducing the 21s/case solver bottleneck.

- **Accelerating LLM Inference with Lossless Speculative Decoding** (ICML 2025) - https://icml.cc/virtual/2025/poster/44892
  Why it matters: Speculative decoding for heterogeneous token distributions with lossless speedup. Draft-and-verify enables batch generation with quality guarantees.
  Carnot use: Architecture guidance for pipelining PredictiveVerifier with main generation. A small verifier draft model could propose violations while the main model generates, hiding verification latency. Relevant to the batched inference improvements planned for the DualGPURunner harness.

### Online Constraint Learning and Repair

- **Learning to Solve and Verify: Sol-Ver Self-Play Framework** (arXiv 2502.14948) - https://arxiv.org/abs/2502.14948
  Why it matters: Jointly improving code generation (19.63% relative improvement) and test generation (17.49%) without human annotations via quality-gated mutual verification. Error accumulation prevention via bidirectional quality checks.
  Carnot use: Template for the constraint addition module improvement — mutual verification between constraint templates and live inference tokens prevents the "too liberal matching" failure seen in Exp 256. Already cited in prior milestones but its quality-gating mechanism is directly actionable.

- **Towards Efficient Constraint Handling in Neural Solvers** (arXiv 2602.16012) - https://arxiv.org/abs/2602.16012
  Why it matters: Learning-based feasibility refinement that refines infeasible solutions in a few post-construction steps while preserving optimality. General, simple, efficient neural constraint handling.
  Carnot use: Online constraint refinement applicable to the constraint_addition module. Rather than adding entirely new templates, this suggests refining existing templates toward feasibility on live inference outputs — complementary to domain-specific token pattern extraction.

### Hardware Acceleration: Ising Machines and NPU

- **All-to-All Reconfigurability with Sparse and Higher-Order Ising Machines** (Nature Communications 2024) - https://www.nature.com/articles/s41467-024-53270-w
  Why it matters: Sparse Ising machines operate at constant frequency on ASIC/FPGA with vanishing overhead for inherently sparse problems. Achieves accurate ground states on 80×80 lattices (6,400 spins) via multi-FPGA cluster.
  Carnot use: Confirms that sparse constraint graph formulation (from Scalable Connectivity paper) is the right design for KV260 bring-up. Constraint checking graphs are naturally sparse — this is a signal to shape future FPGA workloads as sparse Ising problems rather than dense fully-connected ones.

- **Predicting Sampling Advantage of Stochastic Ising Machines** (arXiv 2504.18359) - https://arxiv.org/abs/2504.18359
  Why it matters: Framework for predicting when stochastic Ising machines offer speedup over classical sampling. Identifies problem classes where Ising hardware wins and where classical MCMC is competitive.
  Carnot use: Theoretical guide for which constraint satisfaction workloads should be routed to KV260/TSU hardware vs. CPU MCMC. The PredictiveVerifier's sampling-based gate may or may not benefit from Ising acceleration — this framework provides the analysis tools to predict before investing in hardware integration.

## Revalidation Sweep Insights — Exp 271-280 (2026-04-14)

**CRITICAL FINDING FOR EXTRACTION:** Exp 279 (adversarial semantic grounding) definitively
scoped the current semantic grounding capability:
- **100% stale-answer detection** (model reuses original numbers on a swapped question)
- **0% fresh-wrong detection** (model computes a new wrong answer with correct question numbers)
- **20% FP rate** on correct originals
- **+40pp lift** on stale variants

**This is actionable.** The Apple adversarial GSM8K (arXiv 2410.05229) number-swap variant
generates stale errors: a model that memorized the original answer will use original numbers on
the swapped question. Semantic grounding should achieve **high recall on number-swap variants**
because this is exactly the stale-answer pattern it detects at 100%.

**Prediction for milestone 2026.04.21:** Running Apple adversarial with semantic grounding
should show verify-repair improvement LARGER on number-swap than on standard GSM8K.
This is the first data-driven prediction made from a prior confirmed result.

**What the extraction-free path covers:**
- Spilled energy (arXiv 2602.18671): covers uncertain outputs (high logit/output energy gap)
- AR-EBM lookahead energy (arXiv 2512.15605): covers continuation-level incoherence
- Semantic energy (arXiv 2508.14496): covers confident-but-wrong outputs (low entropy, wrong answer)
- Together: cover the classes that semantic grounding misses (fresh-wrong, uncertain errors)

**Confirmed capabilities from revalidation sweep:**
- Global consistency: 100% contradiction detection on live multi-turn chains (Exp 271, CONFIRMED)
- Agent rollback: 100% success on live Gemma4 workflows (Exp 273, CONFIRMED)
- Semantic grounding stale detection: 100% (Exp 279, CONFIRMED + SCOPED)
- Code PBT verify-repair: +3pp on HumanEval (Exp 226-227, pre-revalidation, CONFIRMED)

### LoRA Continual Learning for Constraint Models — arXiv 2504.13407
- **What:** Orthogonal LoRA tuning with critical parameter freezing prevents catastrophic
  forgetting in continual learning settings. Simple modification to LoRA that preserves
  pre-task performance while adapting to new tasks.
- **Relevance:** Multi-turn agentic verification requires accumulating constraint knowledge
  across sessions without forgetting domain-specific constraint patterns. The orthogonal
  LoRA approach could apply to Carnot's constraint model updates in the self-learning loop.
- **When to pursue:** When Tier 2 constraint memory is proven insufficient and a parametric
  approach to constraint retention is needed.

### Πnet — Hard-Constrained NNs via Orthogonal Projection — arXiv 2508.10480
- **What:** Output layer using operator splitting to guarantee convex constraint satisfaction.
  Maps any unconstrained output onto the feasible constraint set via orthogonal projection.
- **Relevance:** More principled than Langevin repair in VerifyRepairPipeline — would guarantee
  constraint satisfaction rather than just reduce energy. Could replace the "random restart"
  fallback strategy.
- **When to pursue:** Repair pipeline improvement milestone. Add ProjectionRepair strategy
  alongside existing Langevin repair.

### T-SKM-Net — Neural Constraint Satisfaction via Kaczmarz-Motzkin — arXiv 2512.10461
- **What:** Trainable neural network framework for linear constraint satisfaction using a
  differentiable Sampling Kaczmarz-Motzkin method. Handles large-scale linear inequality systems
  with favorable convergence, eliminates non-differentiable argmax operations.
- **Relevance:** Could serve as an alternative to Ising for linear constraint checking —
  faster convergence on purely linear systems (e.g., arithmetic bounds), while Ising remains
  the tool for combinatorial/SAT-style constraints.
- **When to pursue:** When the constraint types are predominantly linear inequalities.

### Sol-Ver Self-Play — Mutual Verification for Constraint Templates — arXiv 2502.14948
- **What:** Jointly improves code generation and test generation without human annotations
  via quality-gated mutual verification (sol verifies ver's tests, ver verifies sol's code).
  19.63% relative improvement on code, 17.49% on tests.
- **Relevance:** Template for the constraint addition module — mutual verification between
  constraint templates and live inference tokens prevents liberal matching failures.
  The quality-gating mechanism is directly applicable to Exp 141 (constraint generation from memory).
- **When to pursue:** When rebuilding constraint generation pipeline.

### Efficient Constraint Handling in Neural Solvers — arXiv 2602.16012
- **What:** Learning-based feasibility refinement that refines infeasible solutions post-construction.
  Simple, efficient, general — works across multiple constraint types.
- **Relevance:** Online constraint refinement applicable to VerifyRepairPipeline's repair step.
  Rather than always restarting from scratch, refine the near-feasible LLM output toward
  feasibility using learned refinement operators.
- **When to pursue:** Repair quality improvement milestone.

## ArXiv Scan — Planning for Milestone 2026.04.21 (2026-04-14)

Queries: EBM verification/reasoning, hallucination detection EBM energy, Ising constraint ML,
KAN FPGA hardware, constrained decoding LLM, continual learning constraints, LLM formal verification,
thermodynamic computing Ising FPGA

### EDLM — Energy-Based Diffusion Language Models (ICLR 2025)
- **ArXiv:** [2410.21357](https://arxiv.org/abs/2410.21357) (2024-10, ICLR 2025)
- **What:** Full sequence-level EBM operating at each diffusion step using parallel importance
  sampling. Achieves 1.3× sampling speedup over existing diffusion LMs and approaches autoregressive
  perplexity. Provides a blueprint for constrained generation that scores entire candidate sequences.
- **Relevance to Carnot:** Maps to Carnot's Boltzmann (large) tier. The sequence-level EBM
  formulation enables constrained generation over entire output sequences, not just token-by-token.
  Relevant for the long-term guided decoding path (Kona parity).
- **When to pursue:** Boltzmann tier redesign milestone. Add sequence-level EBM scoring as an
  alternative to token-level energy in the guided decoding adapter.

### Emergent Formal Verification — Z3 SMT Across 6 Domains (arXiv 2603.21149)
- **ArXiv:** [2603.21149](https://arxiv.org/abs/2603.21149) (2026-03)
- **What:** Documents an AI system (SUBSTRATE S3) that independently discovered Z3 SMT-based
  verification across LLM code safety, agent tool safety, reasoning validation, CLI commands,
  hardware assembly, and smart contracts. Achieved 100% classification accuracy on 181 test cases.
- **Relevance to Carnot:** Provides the strongest existence proof that EBM-guided reasoning
  naturally converges on formal verification as a correctness signal. Six-domain coverage
  suggests Z3 can be a universal hard constraint layer alongside Carnot's soft energy scoring.
  The code + agent tool + reasoning domains all overlap with Carnot's current scope.
- **When to pursue:** Z3 SMT constraint extraction milestone. Use as existence proof that
  Z3-backed verification generalizes across the domains Carnot targets.

### Loop Invariant Generation via LLM + Z3 (arXiv 2508.00419)
- **ArXiv:** [2508.00419](https://arxiv.org/abs/2508.00419) (2025-08)
- **What:** Integrates O1/O3-mini with Z3 in a generate-and-check loop, achieving 100% on
  Code2Inv (133/133), surpassing prior best of 107/133. Only 1-2 LLM calls per task,
  14-55 seconds per instance.
- **Relevance to Carnot:** Demonstrates the efficient LLM+SMT loop architecture that Carnot's
  constraint verification pipeline should adopt. The minimal call count (1-2 per task) is
  directly achievable with the FormalClaimVerifier's current routing approach.
- **When to pursue:** Z3 constraint extraction milestone. Use as blueprint for LLM+SMT call efficiency.

### Ising Inverse Temperature Learning under Hard Constraints (arXiv 2509.20993)
- **ArXiv:** [2509.20993](https://arxiv.org/abs/2509.20993) (2025-09, revised 2026-02)
- **What:** Estimates inverse temperature β for truncated Ising models from a single sample using
  pseudolikelihood maximization, achieving O(Δ³/√n)-consistency. Constraints expressed as k-SAT
  formulas, enabling interoperability with SMT-style constraint specification.
- **Relevance to Carnot:** Direct application to Ising tier parameter estimation. The k-SAT
  constraint expression enables interoperability with FormalClaimVerifier's SMT-style constraints.
  Single-sample β estimation could enable online calibration of the Ising sampler per query.
- **When to pursue:** Ising parameter learning milestone. Single-sample β estimation is cheap
  enough to run at inference time.

### Mpemba-Effect Langevin Initialization for Faster MCMC (arXiv 2603.24183)
- **ArXiv:** [2603.24183](https://arxiv.org/abs/2603.24183) (2026-03)
- **What:** Exploits the Mpemba effect to suppress slow relaxation modes in Langevin-dynamics
  thermodynamic computing. Digitally computed optimal initializations dramatically reduce
  time-to-equilibration.
- **Relevance to Carnot:** Directly applicable to Carnot's Langevin-based repair in
  VerifyRepairPipeline. The initialization trick can substantially cut MCMC wall-clock time
  for the continuous relaxation repair path (Exp 64). Also relevant to FPGA Ising bring-up:
  optimal initialization reduces time to first valid sample.
- **When to pursue:** Sampling performance milestone. Add as initialization option to
  ParallelIsingSampler and FpgaBackend.

### SciDC — Scientific Knowledge-Driven Decoding Constraints (arXiv 2604.06603)
- **ArXiv:** [2604.06603](https://arxiv.org/abs/2604.06603) (2026-04)
- **What:** Automatic framework converting flexible scientific knowledge into multi-layered
  formalized decoding constraints. Qwen3-14B + SciDC scores 86.46% on LegalBench, surpassing
  Claude Sonnet (86.17%) and GPT-4 (85.11%).
- **Relevance to Carnot:** Demonstrates that externally supplied constraint specs can be compiled
  into decoding-time enforcement — the guided decoding path Carnot is targeting. The LegalBench
  result also provides a concrete benchmark target for Carnot's constrained generation approach.
- **When to pursue:** Guided decoding milestone. Use LegalBench as evaluation benchmark.

---

## New Papers — Filed 2026-04-15 (Milestone v30 Planning)

### VERGE — Formal Refinement and Guidance Engine for LLM Verification (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2601.20055 — "VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning" (2026-01)
- **What:** Combines LLMs with SMT solvers (Z3) in an iterative refinement loop. Instead of
  one-shot Z3 checking, VERGE iterates: detect which assertion failed, ask the LLM to fix
  that specific step, re-verify until SAT or max iterations reached. Achieved near-perfect
  accuracy on multi-step math benchmarks via this targeted repair loop.
- **Relevance to Carnot:** Directly extends Carnot's NL2Z3Extractor (Exp 310) and Z3-gated
  repair (Exp 312). The iterative refinement design is the natural next step: instead of
  flagging the whole response as violated, identify which atomic step failed Z3 and prompt
  the LLM to repair only that step. This is more surgical than Carnot's current whole-response
  repair approach and should reduce false positives dramatically.
- **When to pursue:** Next milestone (2026.04.24). Implement VERGE-style iterative loop on top
  of NL2Z3Extractor + Z3GatedRepair.

### CRV — Chain-of-Thought Circuit Verification (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2510.09312 — "Verifying Chain-of-Thought Reasoning via Its Computational Graph" (2025-10)
- **What:** Extracts the computational dependency graph (circuit) from a CoT response — which
  intermediate values depend on which others. Uses structural fingerprints of this graph to
  detect reasoning errors. A "broken" circuit (where a downstream value doesn't follow from
  upstream values) indicates an error. Model-agnostic, no external KB required.
- **Relevance to Carnot:** Complementary to Z3 (which checks arithmetic consistency) — CRV
  checks STRUCTURAL consistency of the reasoning chain. Combined with NL2Z3Extractor: Z3
  checks arithmetic, CRV checks logical flow. Together they cover both types of IT model errors:
  arithmetic mistakes (Z3) and reasoning chain errors (CRV). The circuit extraction maps
  naturally onto Carnot's ConstraintIR — each node is a constraint.
- **When to pursue:** Next milestone. Implement CRV-style CoT graph extractor as a new
  ConstraintExtractor variant: CoTCircuitVerifier.

### Typed CoT — Curry-Howard Framework for LLM Verification (RESEARCH FRONTIER)
- **Paper:** arxiv.org/abs/2510.01069 — "Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning" (2025-10)
- **What:** Maps informal CoT traces to formally typed proof structures using the Curry-Howard
  correspondence (proofs = programs). Each reasoning step must have a type; a type mismatch
  indicates an error. Enables computational verification of reasoning faithfulness without
  a domain-specific KB.
- **Relevance to Carnot:** Provides a formal type-theoretic foundation for constraint
  verification. The type system could be used to validate that each reasoning step's output
  type matches the input type expected by the next step (e.g., "a number" → "a count" →
  "a percentage"). Long-term: integrate into Carnot's ConstraintIR as a type-checking layer.
- **When to pursue:** Research frontier. Study for Kona parity milestone. Near-term: may inform
  CRV-style CoTCircuitVerifier design.

### Solver-Aided Agent Policy Compliance (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2603.20449 — "Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents" (2026-03)
- **What:** Uses SMT solvers (Z3) to enforce policy constraints in LLM agent tool calls at
  runtime. Intercepts tool invocations, checks that parameters satisfy policy assertions, and
  blocks non-compliant calls. Prevents agents from taking harmful actions that satisfy surface-
  level instructions but violate deeper constraints.
- **Relevance to Carnot:** Operational validation of Carnot's multi-turn agentic verification
  (Goal #4). This paper shows SMT-based constraint enforcement for agent tool calls is
  production-viable. Connects to Carnot's ConstraintStateMachine (Exp 125) and
  AgentRollback (Exp 126) — the policy compliance framing is more principled than
  Carnot's current violation-triggered rollback. Also: provides a new benchmark domain
  (agent tool calls) for evaluating Carnot's constraint extraction.
- **When to pursue:** Next milestone — add agent policy compliance as a benchmark mode in
  the extractor benchmark. Long-term: wire into VerifyRepairPipeline as a tool-call gate.

### EBM Reward Models — Energy-Based LLM Alignment (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2504.13134 — "Energy-Based Reward Models for Robust Language Model Alignment" (2025-04)
- **What:** Proposes EBMs explicitly for reward modeling in LLM alignment. Key insight:
  EBMs' partition function acts as a natural uncertainty quantifier — high partition function
  variance = uncertain reward = lower confidence in that training signal. Demonstrated more
  robust alignment than scalar reward models on OOD inputs.
- **Relevance to Carnot:** Validates the EBM-for-evaluation architectural direction. The
  partition function uncertainty idea maps onto Carnot's energy landscape — high energy
  variance across Gibbs samples = uncertain constraint satisfaction = lower confidence
  in repair. Could improve Carnot's ConfidenceVerifier by using partition function variance
  as the confidence signal rather than raw energy.
- **When to pursue:** Constraint precision milestone. Experiment: replace ConfidenceVerifier's
  scalar energy with partition function variance. Requires computing multiple Ising samples.

### ATLAS — Continual Learning for Deployed Agents (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2511.01093 — "Continual Learning, Not Training: Online Adaptation For Agents" (2025-11)
- **What:** Dual-agent system (ATLAS) with persistent learning memory for gradient-free online
  adaptation. One agent handles current queries; a second "learning agent" distills experience
  into a persistent memory store (similar to Carnot's CaseMemory). The key contribution:
  selective memory consolidation — not all experience is worth saving, only "surprising" or
  "high-contrast" interactions.
- **Relevance to Carnot:** Directly applicable to Tier 2 self-learning (constraint memory /
  Trace2Skill). The selective memory consolidation idea addresses a gap in Carnot's
  CaseMemory (Exp 222) — currently all traces are stored, but only high-contrast cases
  (where verification disagreed with LLM confidence) should be retained. This would reduce
  memory footprint and improve per-pattern precision.
- **When to pursue:** Self-learning architecture milestone. Add selective consolidation to
  CaseMemory: only store traces where verified_violation != model_confidence_direction.
  Target: reduce CaseMemory size by 60% while maintaining or improving pattern precision.

---

## New References — Added 2026-04-16 (Milestone 2026.04.28 Planning)

### Physical Analog KAN — Hardware-Native Energy Tier (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2602.07518 — "Physical Analog Kolmogorov-Arnold Networks based on
  Reconfigurable Nonlinear-Processing Units" (Feb 2026)
- **What:** Implements KAN spline activations in silicon using Reconfigurable Nonlinear-Processing
  Units (RNPUs) — multi-terminal nanoscale devices whose input-output characteristics are tuned via
  control voltages. System-level estimates: ~250 pJ/inference, ~600 ns latency. 10^2–10^3x energy
  reduction vs digital MLP at equivalent approximation error.
- **Relevance to Carnot:**
  1. CIKAN's constraint boundaries (ConstraintBoundary dataclass) map DIRECTLY to RNPU saturation
     points. Each constraint boundary becomes one RNPU saturation voltage — hardware cannot output
     configurations that cross the boundary. This is not a software approximation; it is a physical
     constraint.
  2. The 250 pJ/inference target makes constraint verification orders of magnitude cheaper than
     GPU inference. FPGA LUT-based splines are the intermediate step before silicon RNPU.
  3. Phase 3 vision: CIKAN energy tier running on an aKAN chip, 100x more efficient than Ising on FPGA.
- **FPGA path:** Spline LUTs in FPGA → aKAN silicon → TSU hardware. Each step faster and cheaper.
- **When to pursue:** After CIKAN is working (Exp 378). Add hardware compilation analysis: map each
  ConstraintBoundary to an FPGA LUT saturation constraint. Document the silicon path.

### BiKA — Ultra Lightweight KAN Hardware Accelerator (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2602.23455 — "BiKA: Kolmogorov-Arnold-Network-inspired Ultra Lightweight
  Neural Network Hardware Accelerator" (Feb 2026)
- **What:** KAN architecture variant targeting extreme hardware efficiency. Binarized spline
  activations reduce memory bandwidth and multiply-accumulate operations to near-zero.
- **Relevance to Carnot:** Alternative hardware path for the KAN energy tier at the edge. If full-
  precision KAN splines are too expensive for the KV260 FPGA, BiKA's binarized variant could be
  ported. Also relevant for NPU deployment of the JEPA predictor (Tier 3).
- **When to pursue:** When implementing the FPGA KAN energy tier or NPU deployment of JEPA.

### JitRL — Just-In-Time RL, Continual Learning Without Gradient Updates (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2601.18510 — "Just-In-Time Reinforcement Learning: Continual Learning
  in LLM Agents Without Gradient Updates" (Jan 2026)
- **What:** JitRL maintains a dynamic non-parametric memory of (state, action, advantage) triples.
  At inference time, retrieves relevant triples and uses them to directly modulate output logits
  — equivalent to adding a learned offset to the prior policy without any gradient computation.
  Proven to be the closed-form solution to the KL-constrained policy optimization objective.
  Outperforms fine-tuning methods at 30x lower cost.
- **Relevance to Carnot:**
  1. Carnot's Tier 1 self-learning (constraint reweighting) was proven ineffective in Exp 134 because
     simply reweighting existing constraints doesn't change behavior enough. JitRL suggests a better
     approach: maintain a memory of (constraint_type, question_type, outcome) triples and at inference
     time, retrieve the most similar prior constraint invocations to MODULATE the repair threshold —
     not reweight constraints, but change the decision boundary based on what worked before.
  2. The "no gradient updates" property is critical — constraint memory should be instant-update,
     not a training phase. JitRL proves this is theoretically optimal under KL constraints.
  3. Hardware path: the non-parametric memory fits on CPU (just a key-value store). The logit
     modulation is a simple dot product — sub-microsecond on CPU, sub-nanosecond on FPGA.
- **When to pursue:** Next milestone. Implement JitRL-style constraint logit modulation as an
  upgrade to Tier 1 online learning. Replace counter-based reweighting with memory-retrieval
  modulation.

### Ising Machine ↔ Neural Network Correspondence (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2511.00746 — "Correspondence Between Ising Machines and Neural Networks"
  (Nov 2025)
- **What:** Establishes a systematic bijection between Ising machines (Hopfield networks) and
  feedforward neural networks, with a practical method to compile trained neural networks into
  Ising hardware. Runs feed-forward NNs on Ising-type hardware accelerators.
- **Relevance to Carnot:**
  1. The EORM (EnergyRewardModel, Exp 346) is a small transformer. If this correspondence holds for
     transformer-like architectures, EORM could be compiled to FPGA Ising hardware, eliminating GPU
     dependency for the EORM gate in the three-tier pipeline.
  2. The JEPA predictor (a small binary classifier) is an even simpler candidate — a fully-connected
     2-layer network may compile directly to Ising.
  3. Validates our architecture: by designing small constraint models (JEPA, EORM), we're already
     building in the direction of Ising-hardware-compilable networks.
- **When to pursue:** Hardware acceleration milestone. After JEPA predictor and EORM are stable,
  experiment with compiling them to Ising representation for KV260 FPGA execution.

### Adaptive Weighted Rejection Sampling — Constrained Generation (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2504.05410 — "Fast Controlled Generation from Language Models with
  Adaptive Weighted Rejection Sampling" (April 2026)
- **What:** Adaptive rejection sampling algorithm for constrained LLM generation that requires
  orders of magnitude fewer constraint evaluations than naive rejection sampling. The key insight:
  adapt the sampling distribution to concentrate probability mass near constraint boundaries.
- **Relevance to Carnot:** Directly applies to energy-guided decoding (Exp 110, Exp 138). The
  adaptive sampling approach could replace Carnot's current Langevin dynamics sampler in the
  energy-guided decoding path, making constrained generation faster. Pairs with the JEPA predictor:
  JEPA identifies WHERE violations are likely; adaptive rejection sampling avoids them efficiently.
- **When to pursue:** Energy-guided decoding milestone. Replace current decoding sampler with
  adaptive weighted rejection sampling. Measure tokens/second improvement.

### AutoResearch-RL — Perpetual Self-Evaluating RL for Architecture Discovery (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2603.07300 — "AutoResearch-RL: Perpetual Self-Evaluating RL Agents
  for Autonomous Neural Architecture Discovery" (March 2026, Jain et al.)
- **What:** An RL agent that proposes code modifications to neural architectures, evaluates
  validation loss as a fitness signal, and applies PPO over ~300 iterations without human
  involvement. The perpetual self-evaluation loop uses val-bpb (validation bits-per-byte) as
  its reward signal. Achieves architecture improvements comparable to hand-tuned baselines.
  Note: withdrawn from arXiv for policy reasons but publicly referenced in multiple surveys.
- **Relevance to Carnot:** Direct analog to Carnot's autoresearch conductor loop. The pattern
  is identical: propose experiment -> execute -> observe metric -> update policy. Carnot's conductor
  uses energy-reduction as its fitness signal; AutoResearch-RL uses val-bpb. The PPO-over-code
  paradigm confirms that code-level RL for architecture discovery is viable at small scale.
  Carnot's JitRL-based self-learning (Tier 1) is the non-gradient equivalent: instead of PPO
  weight updates, use non-parametric logit modulation from the energy memory. AutoResearch-RL's
  evaluation harness (spawn -> train -> eval -> score -> record) is the exact structure of
  Carnot's conductor scripts.
- **When to pursue:** FR-11 (Autonomous Self-Learning Loop) confirmation milestone (Exp 415+).
  When validating the self-learning relay, use the val-bpb metric pattern as a principled
  fitness signal for comparing constraint verification quality across JitRL iterations.

### REGREACT — Regulatory Constraint Extraction Pipeline (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2604.12054 — "REGREACT: Self-Correcting Multi-Agent Pipelines for
  Structured Regulatory Information Extraction" (April 2026)
- **What:** Multi-agent pipeline for extracting structured information from regulatory documents.
  Agents decompose regulatory text into logical rules, verify cross-rule consistency, and produce
  machine-readable compliance assertions. Self-correcting: agents flag contradictions and resolve
  them via a negotiation loop.
- **Relevance to Carnot:** Validates the compliance checker product direction (Tier B in
  research-program.md). The multi-agent extraction pattern (decompose → assert → negotiate → resolve)
  maps directly onto Carnot's verify-repair loop. The key insight: regulatory compliance is a
  STRUCTURED constraint domain — rules are formal, assertions are checkable, and violations are
  deterministic. This is the ideal domain for Carnot's constraint verification approach.
  Finance (Basel III/IV), healthcare (HIPAA), and legal (GDPR) all have machine-readable rule sets
  that could be formalized as Carnot constraint templates.
- **When to pursue:** Compliance checker product milestone. Start with a simple domain:
  encode 10 HIPAA rules as ConstraintTemplates, verify GPT/Gemma outputs for compliance.

### Verifiable Process Reward Models (VPRM) — Step Verification via Deterministic Rules (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2601.17223 — "Beyond Outcome Verification: Verifiable Process Reward Models
  for Structured Reasoning" (January 2026)
- **What:** Trains a PRM where EVERY intermediate reasoning step is checked by a deterministic,
  rule-based verifier — not a neural judge. The reward is fully computable via rule-based checks,
  making all components of the reasoning trajectory verifiable. Applied to medical evidence synthesis;
  achieves up to 20% higher F1 than state-of-the-art models. Avoids neural judge opacity, bias, and
  reward hacking.
- **Relevance to Carnot:** Carnot's Ising/KAN constraint verification IS a VPRM. The energy function
  scores each reasoning step deterministically. This paper provides the formal framing Carnot was
  missing: we should train and describe our pipeline as a VPRM, not just a "verifier." The
  deterministic-rules framing also validates our Z3-based extraction (arXiv 2601.17789) as producing
  the verifiable labels needed to train a VPRM. Carnot's advantage: our rules are hardware-acceleratable
  (Ising/KAN vs neural scoring).
- **When to pursue:** Next milestone. Implement VPRM training loop: use Z3-verified step labels from
  FOVER approach (arXiv 2505.15960) to train IsingEBM as a step-level process reward model.

### FOVER — Formally Verified Training Data for Process Reward Models (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2505.15960 — "Generalizable Process Reward Models via Formally Verified
  Training Data" (May 2025)
- **What:** Synthesizes PRM training data with accurate step-level error labels automatically annotated
  by formal verification tools (Z3 and Isabelle), without human annotation. Generates 80K steps in
  reasoning traces with error labels from Llama 3.1 8B and Qwen 2.5 7B. PRMs trained on FOVER data
  generalize significantly better than those trained on human-annotated data.
- **Relevance to Carnot:** FOVER is the upstream data pipeline for VPRM training. Carnot already has
  Z3 formalization (LLMz3Formalizer, Exp 357) and step extraction (CIKANEnergy, Exp 405). The FOVER
  approach: take GSM8K responses, parse CoT steps, send each step to Z3, annotate correct/incorrect,
  use those labels to train IsingEBM as a step-level PRM. This closes the loop: Z3 generates labels,
  Ising learns from them, Ising checks future steps without Z3.
- **When to pursue:** Immediately after live GPU benchmarks produce real CoT traces. FOVER's Z3
  annotation pipeline is already partially built (Exp 357); extend it to step-level annotation.

### ThinkPRM — Process Reward Models That Generate Verification CoT (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2504.16828 — "Process Reward Models That Think" (April 2025, ICLR 2026)
- **What:** Trains a PRM that GENERATES a verification chain-of-thought for every step it scores,
  rather than producing a scalar score. Surpasses discriminative verifiers trained on full PRM800K
  by 8% and outperforms LLM-as-a-Judge by 7.2% on ProcessBench. Data-efficient: the CoT verifier
  learns general reasoning patterns that transfer across domains.
- **Relevance to Carnot:** An alternative to energy-based step scoring. ThinkPRM's verification CoT
  could be used to GENERATE ground-truth step labels for training Carnot's IsingEBM. The model-generated
  verification reasoning could replace human annotation AND Z3 annotation for domains where formal
  verification is too coarse. Particularly valuable for semantic reasoning steps (not just arithmetic).
- **When to pursue:** After FOVER/VPRM experiments. Use ThinkPRM-style verification CoT to extend
  label generation beyond Z3's arithmetic domain to natural language reasoning steps.

### AMD NPU Bare-Metal IRON Toolflow — Skip VitisAI (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2504.03083 — "Unlocking the AMD Neural Processing Unit for ML Training
  on the Client Using Bare-Metal-Programming Tools" (April 2025)
- **What:** Uses IRON (Infrastructure for Research on NPUs) bare-metal toolflow to program AMD XDNA
  NPU directly, bypassing the VitisAI EP (which requires custom onnxruntime build and blocked by
  missing ninja/openblas). Achieves 2.8x GEMM speedup and 1.7x end-to-end speedup for GPT-2 fine-tuning.
  Works with Python 3.11+, standard packages. Does NOT require VitisAI EP or AMD's custom onnxruntime.
- **Relevance to Carnot:** This UNBLOCKS the NPU experiments (Exps 292, 303, 314, 335) that have been
  blocked for 5 milestones by the VitisAI EP build requirement. IRON toolflow provides direct NPU
  programming without the VitisAI dependency chain. The AMD XDNA NPU in our Ryzen AI 9 HX 370 is
  accessible via this path. Target: run JEPA predictor model on NPU while LLM runs on GPU (Tier 3
  self-learning).
- **When to pursue:** Next milestone. Implement NPU backend using IRON toolflow. Install: pip install
  mlir-aie (IRON dependency). No VitisAI EP needed. See github.com/Xilinx/mlir-aie for IRON tools.

### Digitally Optimized Initializations for Fast Thermodynamic Computing (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2603.24183 — "Digitally Optimized Initializations for Fast Thermodynamic
  Computing" (March 2026)
- **What:** Pre-computes near-optimal spin configurations as starting points for thermodynamic sampling,
  dramatically reducing the number of thermodynamic steps needed to reach the ground state. The
  initialization is computed digitally (fast, exact) and handed to the hardware sampler as a warm start.
  Results: 5-10x reduction in convergence time on Ising problems up to 10,000 spins.
- **Relevance to Carnot:** Directly applicable to Carnot's Ising sampler and planned FPGA/TSU hardware.
  Currently, every Carnot sampling run starts from a random spin configuration. Adding a digital
  pre-optimizer (deterministic heuristic or greedy initialization) would give the same speedup benefit
  for both CPU and FPGA sampling. Pairs with the KV260 FPGA backend (Exp 289): warm-start the FPGA
  sampler with a digitally-computed near-optimal configuration.
- **When to pursue:** FPGA hardware milestone. Implement DigitalInitializer that computes warm-start
  spin configurations via greedy descent, then hands to Ising sampler. Benchmark vs random init.

### Self-Certainty Best-of-N — Reward-Free Output Selection (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2502.18581 — "Scalable Best-of-N Selection for Large Language Models via
  Self-Certainty" (February 2025)
- **What:** Self-certainty measures divergence of predicted token distribution from uniform, used as
  a proxy for response quality WITHOUT an external reward model. Best-of-N selection using self-certainty
  is computationally cheap (no additional model call) and competitive with reward-model-based selection.
- **Relevance to Carnot:** The candidate ranker (score_candidates MCP tool) currently uses energy
  scoring (low energy = higher rank). Self-certainty offers a COMPLEMENTARY signal: select for both
  low energy (constraint satisfaction) AND high certainty (confident generation). Combining Carnot's
  energy score with self-certainty could beat either alone for the candidate ranking product (Tier A,
  research-program.md).
- **When to pursue:** Candidate ranker improvement milestone. Add SelfCertaintyScorer alongside
  energy scorer in score_candidates. Benchmark combined score vs energy-only and certainty-only.

### GPU-Accelerated Oscillator Ising Machine (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2505.22631 — "GPU-Accelerated Simulated Oscillator Ising/Potts Machine
  Solving Combinatorial Optimization Problems" (May 2025)
- **What:** GPU-native oscillator-based Ising solver using custom CUDA kernels for combinatorial
  optimization. Oscillator Ising machines map spin dynamics to phase synchronization; GPU acceleration
  enables high parallelism. Demonstrated practical speedups on standard benchmark constraint problems.
- **Relevance to Carnot:** Carnot's parallel Ising sampler (183x over thrml) runs on CPU. Porting the
  inner loop to GPU CUDA kernels using oscillator-style phase synchronization could achieve another
  order of magnitude speedup. The FpgaBackend (Exp 289) abstraction already supports swappable
  backends; a GPUIsingBackend would be the highest-impact sampler upgrade before FPGA hardware ships.
- **When to pursue:** Next milestone. Implement GPUIsingBackend using oscillator phase dynamics on CUDA.
  Benchmark vs CPU ParallelIsingSampler on 100/500/1000-spin problems.

### Kolmogorov-Arnold Energy Models (KAEM) — Exact Sampling (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2506.14167 — "Kolmogorov-Arnold Energy Models: Fast, Interpretable
  Generative Modeling" (June 2025)
- **What:** KAEM imposes univariate latent structure (from KA Representation Theorem) enabling EXACT
  inference via inverse transform sampling — no MCMC required. Bridges simple priors and expressive
  iterative samplers. Fast, interpretable, and closed-form invertible for the energy model.
- **Relevance:** Carnot's current KAN energy tier (Exp 96) uses iterative Ising sampling. KAEM's
  inverse transform exact sampling eliminates the MCMC inner loop entirely, reducing per-check latency
  from milliseconds to microseconds. This is a major speedup for the verification fast path.
  Pairs with the CIKAN boundary priors (Exp 414) — KAEM structure + CIKAN boundaries = exact inference
  over constrained energy landscapes.
- **When to pursue:** KAN tier improvement milestone. Implement KAEMEnergy as fast-path KAN variant
  with exact sampling. Compare latency and accuracy vs IsingEBM and standard KANEnergy.

### Differentiable Symbolic Planning with Feasibility Channels (already added above, see DSP entry)

### ThinkPRM — Process Reward Models That Think (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2504.16828 (April 2025, OpenAI)
- **What:** Generative Process Reward Model that verifies reasoning steps via chain-of-thought.
  Instead of a discriminative "is this step correct?" classifier, ThinkPRM generates a short CoT
  (3-5 steps) verifying the candidate step, then produces a correctness label. Achieves SOTA on
  MATH-500 and AIME '24 with only 1% of typical process supervision labels. The key insight:
  thinking about whether a step is correct is more sample-efficient than training a binary classifier.
- **Relevance to Carnot:** Carnot's EORM (Energy-based Output Reward Model) is a discriminative
  verifier — it assigns scores to CoT steps without reasoning about WHY they are wrong. ThinkPRM
  suggests a "CarnotThinkProbe" that generates a brief verification CoT before triggering full
  Ising verification. This pre-filter could catch obvious errors (wrong arithmetic) before the
  Ising sampler runs, reducing full verification calls by 30-50%.
- **Concrete experiment:** Implement CarnotThinkProbe as a fast pre-filter: given an LLM response,
  ask a secondary Qwen3.5-0.8B to generate a 3-step check ("Is the arithmetic correct?"). If the
  check CoT concludes "incorrect," skip Ising and flag as violation immediately (confidence = 0.8).
  Only run Ising for "uncertain" or "correct" secondary verdicts.
- **When to pursue:** Next milestone. Bridges ThinkPRM's CoT verification with Carnot's Ising
  energy model for a fast-path/slow-path verification architecture.

### Boltzmann-GPT — DBM World Models + Language Generation (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2601.17094 (January 2026)
- **What:** Separates world modeling (Deep Boltzmann Machine, energy-based) from language generation
  (frozen GPT decoder). The DBM learns latent structure from data; an adapter projects DBM latent
  samples to LLM embedding space; the frozen LLM renders samples as text. Enables causal
  interventions and coherent long-range generation without modifying the LLM.
  DBM and GPT trained independently; only adapter is jointly trained (small, fast).
- **Relevance to Carnot:** Carnot's IsingEBM is a specialized DBM (discrete, sparse, constraint-
  shaped). Boltzmann-GPT shows that DBM latent samples can be projected into LLM embedding space
  to guide generation. This is the bridge from Carnot's constraint energy landscape to text-space
  repair: when the IsingEBM finds a low-energy (constraint-satisfying) configuration, project
  it into LLM embedding space to generate a corrected response. This replaces the current "ask
  LLM to fix error" repair step with an energy-guided alternative.
- **When to pursue:** Next milestone. Implement a lightweight BoltzmannRepairBridge that takes
  an Ising ground-state configuration and maps it to an LLM steering direction for repair.

### Energy Matching — Unified Flow + EBM Generation (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2504.10612 (October 2025, NeurIPS 2025)
- **What:** Unified framework combining flow matching and energy-based models via a scalar potential
  energy field that guides optimal transport from noise to data. The flow trajectory minimizes energy
  along the path; the energy function's Boltzmann equilibrium serves as the target distribution.
  Generative model: start from noise, follow energy gradient to low-energy (data-like) regions.
- **Relevance to Carnot Phase 3:** The Phase 3 North Star is continuous-latent non-autoregressive
  reasoning (Kona-like). Energy Matching provides the generation mechanism: start from a noise
  vector, follow the constraint energy gradient to reach a constraint-satisfying reasoning state.
  This is the continuous-space analog of Carnot's discrete Ising sampling. The flow matches
  the distribution of valid reasoning states (low energy) without autoregression.
- **When to pursue:** Phase 3 seed experiments. Use Energy Matching trajectory as the sampling
  algorithm for the ContinuousEBM (Exp 435a). Compare convergence vs gradient descent + Langevin.

### Generative Thermodynamic Computing (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2506.15121 (June 2025, Stephen Whitelam, LBL)
- **What:** Proposes generative modeling via natural Langevin dynamics of physical systems under
  thermodynamic equilibrium. Training minimizes heat emission by reversing noising trajectories
  (time-reversal of diffusion). Data synthesis follows thermodynamic evolution from initial
  state to equilibrium. Training objective is the thermodynamic free energy.
- **Relevance to Carnot:** Carnot's parallel Ising sampler uses simulated annealing (synthetic
  temperature schedule). Generative Thermodynamic Computing suggests using physically-motivated
  Langevin dynamics instead of heuristic annealing — the temperature schedule emerges from
  the free energy landscape rather than being hand-tuned. Applicable to the ContinuousEBM (Exp
  435a) and FpgaBackend thermodynamic simulation. Bridges to Extropic TSU (genuine thermodynamic
  hardware).
- **When to pursue:** Phase 3 seed / FPGA milestone. Use thermodynamic Langevin dynamics as the
  sampling algorithm for ContinuousEBM. Compare to SA annealing schedule.

### Process Reward Agents (PRA) — Decoupled Step-Level Verification (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2604.09482 (April 2026, Alibaba DAMO)
- **What:** Process Reward Agents provide domain-grounded, step-wise rewards during generation
  without modifying the base reasoning model. A separate "reward agent" observes each reasoning
  step, assigns a step reward using domain knowledge (medical, legal, math), and feeds signals
  back to guide generation. Achieves 80.8% on MedQA at 4B scale without any policy fine-tuning.
  Fully decoupled — the policy model stays frozen; only the reward agent is domain-specific.
- **Relevance to Carnot:** Carnot's VerifyRepairPipeline is a post-hoc verifier (checks after
  generation). PRA suggests wiring Carnot as a real-time process reward signal during generation.
  The IsingEBM becomes the "reward agent" — it assigns step rewards as the LLM generates each
  reasoning step. This is the step-level guided decoding integration that's been on the roadmap
  (Exp 110, energy-guided decoding). PRA's decoupled architecture means no LLM modification needed.
- **When to pursue:** Guided decoding milestone. Integrate IsingEBM as process reward agent in
  a PRA-style loop for step-by-step guided generation. Compare to Exp 110 energy-guided decoding.

### VPRM — Verifiable Process Reward Models for Structured Reasoning (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2601.17223 (2026-01, Pronesti, Belz, Hou)
- **What:** Introduces Verifiable Process Reward Models (VPRM), a reinforcement-learning framework
  where intermediate reasoning steps are verified by deterministic, rule-based verifiers instead
  of neural judges. Applied to medical evidence synthesis with guideline-defined criteria.
  Results: 20% higher F1 than SOTA, 6.5% higher than verifiable outcome rewards. Reasoning traces
  are more coherent and evidence-grounded. Rule-based verifiers eliminate reward hacking.
- **Relevance:** Direct approach for Carnot's extraction problem on IT models. The ArithmeticExtractor
  regex misses 100% of violations on Gemma4-E4B-it because it matches `a + b = c` patterns that IT
  models don't write. VPRM's fix: replace regex with RULE-BASED verifiers: arithmetic rules
  (addition, multiplication, unit consistency), logical consistency rules. These rules are deterministic,
  transparent, and don't require a trained extractor — they directly formalize what "correct
  arithmetic" means and check each CoT step.
- **Concrete experiment:** VPRMArithmeticVerifier: parse each CoT step for numerical claims using
  LLM formalization, apply Python arithmetic rules (not regex), flag steps where the claimed
  result doesn't match the computed result. Compare detection rate vs ArithmeticExtractor and
  LLMExtractor on GSM8K with Gemma4-E4B-it.
- **When to pursue:** Next milestone (2026.04.34). Implement as Tier 3 step-level verifier.
  Pairs with VeriCoT for neuro-symbolic extraction.

### VeriCoT — Neuro-Symbolic Chain-of-Thought Validation via Logical Consistency (HIGH PRIORITY)
- **Paper:** arxiv.org/abs/2511.04662 (2025-11)
- **What:** VeriCoT formalizes each CoT reasoning step into first-order logic (FOL) and uses
  automated solvers (Z3) to verify logical validity. Identifies premises grounding each argument
  in context, commonsense knowledge, or prior steps. Used for inference-time self-reflection:
  VeriCoT's validity-oriented feedback prompts model to self-correct — 46% relative improvement
  in CoT verification pass rate, 41% relative gains in task accuracy across multiple datasets.
- **Relevance:** This is the extraction approach that works on instruction-tuned models. Instead
  of regex matching, VeriCoT uses LLM-assisted formalization of each CoT step into FOL, then Z3
  verifies consistency. This directly solves the ArithmeticExtractor 0% detection problem on IT
  models: formalize "47 + 28 = 76" as a Z3 assertion, verify it fails, flag the step.
  The 46% improvement in verification pass rate validates this as a production-ready approach.
- **Concrete experiment:** VeriCoTStepValidator: for each CoT step, use secondary Qwen3.5-0.8B
  to extract FOL premises + conclusions, verify with Z3, flag logically inconsistent steps.
  Wire as extraction front-end in VerifyRepairPipeline.
- **When to pursue:** Next milestone (2026.04.34). Implement as primary extraction mechanism for
  IT models, alongside VPRM rule-based verifiers.

### LSEBMCL — Latent Space EBM for Continual Learning (MEDIUM PRIORITY)
- **Paper:** arxiv.org/abs/2501.05495 (2025-01, ICAIIC 2025)
- **What:** Integrates an EBM layer into continual learning for NLP. The EBM acts as a generative
  memory — it samples from the distribution of PREVIOUS tasks when training on new ones, preventing
  catastrophic forgetting. The novel prior integrates continuous latent variables for generation
  and discrete latent variables for structural elements. Achieves SOTA across NLP continual learning.
- **Relevance:** Directly applicable to Carnot's cross-session constraint memory failure (Exp 448:
  no improvement across sessions). The root cause of Exp 448's failure: SessionMemory stored
  constraint templates, but Session 2 couldn't leverage them because no replay mechanism existed.
  LSEBMCL's fix: train a small Ising EBM on constraint violation patterns from Session 1, then
  REPLAY synthetic violations in Session 2 to warm-start the template library. The EBM GENERATES
  the cross-session memory signal rather than just storing it.
- **Concrete experiment:** LSEBMConstraintReplayer: train a small Ising EBM on violation type
  distributions from Session 1 (carry/sign/unit errors). Before Session 2, replay N synthetic
  violations from the EBM to pre-activate constraint templates. Measure if cross-session FP
  reduction improves over Exp 448's baseline (no_improvement).
- **When to pursue:** Next milestone (2026.04.34). Wire into cross-session relay as EBM-native
  memory mechanism.

### Gemma4 Tokenizer Bug — Infinite <unused> Token Loop (CRITICAL INFRASTRUCTURE NOTE)
- **Issue:** github.com/ggml-org/llama.cpp/issues/21516 (April 2026)
- **What:** Gemma4 models (including gemma-4-E4B-it) generate an infinite stream of <unused>
  tokens (token ID 14 = <unused8>) when loaded via llama.cpp backend, producing no valid text.
  Affects both GPU offloading and CPU-only inference. Root cause: tokenizer vocabulary mismatch
  between llama.cpp's Gemma4 tokenizer implementation and the model's expected format.
- **Root cause of RETRO-028:** This is WHY Gemma4-E4B-it returned 0.0 accuracy across all
  benchmarks in milestone 2026.04.33 — the model never produced any valid text. The Ising
  pipeline received empty/garbage responses, producing 0% correct answers.
- **Fix:** Use transformers library directly (AutoModelForCausalLM.from_pretrained) instead of
  any llama.cpp-based backend. The Hugging Face transformers implementation handles Gemma4's
  tokenizer correctly. Alternatively: use tokenizer.chat_template formatting explicitly.
- **Impact on research:** ALL Exp 439/440/441 results showing Gemma4 0% accuracy are
  infrastructure failures, not EBM failures. The model is capable of correct answers — the
  loading path was broken. Re-run with transformers loader for honest results.
- **When to pursue:** Next milestone (2026.04.34), Exp 450 as first priority.

### EBM-CoT — Energy-Based Calibration for Implicit Chain-of-Thought (MEDIUM PRIORITY)
(Previously filed under "Think Consistently, Reason Efficiently" — arXiv:2511.07124.
 Now upgrading to high actionability given EORM+JEPA training on real data (Exp 443).)
- **Update for milestone 2026.04.34:** With JEPA AUC improved from 0.457→0.571 on 57 real
  CoT steps (Exp 443), we now have enough real data to implement EBM-CoT's consistency energy
  formulation as a training objective for EORM. EBM-CoT calibrates the hidden state BEFORE
  generation by running Langevin dynamics on thought embeddings — if we apply this to the EORM
  input encoding, it could improve AUC beyond 0.571 by making the EORM input more "consistent"
  before scoring. Experiment: add Langevin calibration step to EORM's forward pass on hidden
  states from Exp 443's real training data.

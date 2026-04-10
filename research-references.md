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

(Add more papers, arxiv links, and theoretical ideas here as they come up)

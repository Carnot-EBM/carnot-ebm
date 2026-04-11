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


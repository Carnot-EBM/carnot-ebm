# Research References & Future Considerations

Items filed here are technologies, papers, repos, and ideas to consider
in future research milestones. The research conductor and planning agent
should read this file when designing new milestones.

## 2026-05-03 Scan (Milestone 2026.04.93 Planning)

### Spurious Rewards in RLVR: Random Rewards Achieve 73% of GRPO Gains
- **Paper:** arXiv 2506.10947 (2026).
- **What:** RLVR training with GRPO can improve MATH-500 by 21.4pp using randomly-assigned rewards, vs 29.1pp from ground-truth rewards. Approximately 73% of the GRPO performance gain comes from the training *structure* (group sampling, advantage normalization) rather than the reward signal itself.
- **Relevance to Carnot:** Challenges the assumption that improving reward quality (TinyV, GRPO-VPS) will yield proportional gains. The first 73% of Carnot's GRPO improvement may be structural; the remaining 27% is where reward quality actually matters. Implies Latent-GRPO (exp1187: no_delta) may not have failed because of poor masking — it may have failed because the energy reward signal is not in the upper 27%. Recalibrate expectations for GRPO v5: meaningful improvement over v4 (+10pp) would confirm Carnot's energy reward is in the load-bearing 27%.
- **Concrete experiment:** Compare GRPO v5 (energy-reward) vs a GRPO-random-reward ablation on the same training set. If v5 beats random by >3pp, the energy verifier contributes real signal. File as follow-on to exp1195.
- **When to incorporate:** .93 as GRPO v5 interpretation context; .94 for ablation if v5 succeeds.

### LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning
- **Paper:** arXiv 2510.04573 (October 2025).
- **What:** LaDiR (Latent Diffusion Reasoner) constructs a structured latent reasoning space using a VAE that encodes reasoning steps into thought-token blocks, then runs a latent diffusion model over those blocks for iterative refinement. Key finding: latent diffusion over compact thought representations generalizes better than token-level diffusion.
- **Relevance to Carnot:** DoT (exp1186) retired because per-token EBM energy gradients were flat (AUROC=0.5 at all T). LaDiR's approach — encode full reasoning chains into a compact latent, then diffuse over the latent — could replace per-token masking with sequence-level EBM guidance on the VAE latent. The DBAE (Deterministic Bounded Autoencoder) from Phase 3 would be the encoder; the Ising EBM would score latent states. This is a principled redesign of DoT that avoids the token-granularity problem.
- **Concrete experiment:** Redesign DoT using LaDiR pattern: DBAE encodes CoT steps → latent EBM scores → diffusion over latents → decode corrected reasoning chain. Benchmark on FoVer vs failed per-token DoT (exp1171/1186).
- **When to incorporate:** .94+ after DBAE substrate is further validated.

### R-Zero: Self-Evolving Reasoning LLM via Challenger/Solver Co-Evolution
- **Paper:** arXiv 2508.05004 (August 2025).
- **What:** A base model is split into two roles — Challenger (rewarded for generating hard tasks) and Solver (rewarded for solving them). The two roles co-evolve via RL, with the Challenger's difficulty signal bootstrapping the Solver's improvement. No human-curated curriculum required.
- **Relevance to Carnot:** Carnot's Phase 4 active inference (exp1165/1189) needs harder puzzles than synthetic trivially-solvable ones. R-Zero's Challenger mechanism could generate harder ARC-AGI-3 style puzzles where the Challenger is rewarded if BFS is intractable (state-space > 100k) but Phase 4 can solve them. This would create an auto-curriculum that scales puzzle difficulty with Carnot's Phase 4 capability.
- **Concrete experiment:** Implement a simple Challenger model that generates ARC-style 15x15 grids where BFS exceeds 100k states, rewarded for creating Phase-4-solvable-but-BFS-hard puzzles. Run Solver (Phase 4 energy minimization) against Challenger-generated puzzles.
- **When to incorporate:** .94+ after Phase 4 baseline is validated (exp1197 must first show Phase 4 can beat BFS on intractable cases).

### ARC-NCA: Developmental NCA-Based Solutions for ARC-AGI
- **Paper:** arXiv 2505.08778 (May 2025).
- **What:** Neural Cellular Automata (NCA) that learn local developmental rules to solve ARC-AGI tasks without explicit reasoning chains. The NCA evolves a grid state using learned local update rules, converging to the target output.
- **Relevance to Carnot:** Phase 4's active inference model minimizes free energy over a grid state — similar convergence semantics to NCA. ARC-NCA's local update rules are interpretable as constraint-satisfying transformations, which map naturally to Carnot's Ising constraint encoding. Could hybridize: use NCA to generate candidate solutions, Carnot energy to score and select them.
- **When to incorporate:** .94+ as comparative baseline for Phase 4 puzzle results.

### KANtize: Low-Bit Quantization of KAN for Efficient Inference
- **Paper:** arXiv 2603.17230 (March 2026).
- **What:** Explores 4-bit and 8-bit quantization of KAN spline functions for efficient inference. Key finding: 8-bit KAN quantization preserves >99% of full-precision accuracy; 4-bit shows ~2% accuracy degradation but achieves 2x memory reduction. Spline endpoint quantization is more sensitive than interior quantization.
- **Relevance to Carnot:** SOS-KAN verifier (AUROC=0.9902) could be deployed on AMD XDNA NPU or other edge hardware with 4-bit quantization (~2.2MB model size). The 2% AUROC degradation (0.9902 → ~0.9700) would still exceed Carnot's k=5 ensemble target (0.9240 AUROC). Critical for sovereignty claim: constraint verification on consumer hardware.
- **Concrete experiment:** Apply 4-bit/8-bit quantization to SOS-KAN energy verifier. Measure AUROC pre/post quantization. If AUROC stays above 0.97, the quantized model is production-ready for edge deployment.
- **When to incorporate:** Milestone .93 exp1199.

### GRPO-VPS: Step-Level Process Supervision for GRPO
- **Paper:** arXiv 2604.20659 (April 2026) — already in .91 scan.
- **New finding:** Segment-wise process signals computed as change in model's belief in the correct answer across consecutive reasoning segments. Up to 2.6pp improvement on math and 13.7% reasoning-length reduction. Attribute credit to the specific reasoning step that caused the error.
- **Relevance update:** Carnot's CausalReasoningVerifier (Tier 2.7) + Z3MathVerifier (Tier 3) produce step-level signals. Wiring them as GRPO-VPS segment rewards directly assigns credit to the step where a carry-forward error occurred.
- **Concrete experiment:** exp1196 in .93 — wire step-verifiers as GRPO-VPS segment rewards. Compare against GRPO v4 (+10pp) baseline.

## 2026-05-02 Scan (Milestone 2026.04.92 Planning)

### Latent-GRPO: Latent-Space Group Relative Policy Optimization
- **Paper:** arXiv 2604.27998 (April 2026).
- **What:** Novel GRPO variant combining invalid-sample masking (skip reward signal for trivially
  easy samples where all completions score the same) with one-sided noise sampling (add structured
  noise only to incorrect samples, not correct ones). Reports 7.86-point improvement on low-difficulty
  tasks and 3-4x shorter reasoning chains vs standard GRPO.
- **Relevance to Carnot:** GRPO v5 (exp1173/.91) blocked by llama.cpp GPU offload issue. Latent-GRPO's
  invalid-sample masking directly complements TinyV false-negative correction: when energy is ~uniform
  across completions (trivial sample), mask the reward signal — this is orthogonal to TinyV which masks
  when verifier confidence is uncertain. One-sided noise is also complementary: correct samples get no
  noise (energy stays low), incorrect samples get noise to push energy higher. Could apply both
  improvements on top of GRPO v5 architecture.
- **Concrete experiment:** Apply Latent-GRPO invalid-sample masking (skip when energy_std < threshold
  across completions) + one-sided noise to Carnot's GRPO energy training. Compare vs GRPO v4 (+10pp)
  and GRPO v5 baseline. Run standalone without llama.cpp GPU dependency (use CPU for scoring).
- **When to incorporate:** Milestone .92 Phase 5, exp1187.

### GRPO-VPS: GRPO with Verifiable Process Supervision
- **Paper:** arXiv 2604.20659 (April 2026).
- **What:** Addresses credit assignment for intermediate reasoning steps in GRPO. Rather than a single
  outcome reward, computes per-step rewards using a process verifier. Improves on GRPO by attributing
  credit to the specific reasoning step that caused the correct/incorrect outcome.
- **Relevance to Carnot:** Carnot has per-step verifiers: CausalReasoningVerifier (Tier 2.7),
  SymCodeVerifier (Tier 2.5), Z3MathVerifier (Tier 3). These naturally produce step-level verification
  signals. Wiring them as GRPO-VPS process rewards could attribute training signal to the exact step
  where an arithmetic error occurred, rather than blaming the whole chain.
- **Concrete experiment:** Wire CausalReasoningVerifier + Z3MathVerifier as per-step reward signals in
  GRPO. Compare step-level vs outcome-level improvement_over_baseline on held-out GSM8K.
- **When to incorporate:** .93 or later (after GRPO v5 GPU offload issue is resolved).

### Energy-Based Diffusion Language Models
- **Paper:** arXiv 2410.21357 (October 2024, still relevant 2026).
- **What:** EBM-guided diffusion for text generation. Uses energy function to guide the reverse
  diffusion process — high-energy states get higher noise, low-energy states are preserved. Provides
  the theoretical connection between EBM verification and diffusion-based text correction.
- **Relevance to Carnot:** DoT inference (exp1171/.91) found AUROC=0.5 at all T values — the energy
  gradient didn't capture any per-token signal. This paper's EBM-diffusion formulation suggests a
  different approach: instead of computing energy deltas per token (which may be too fine-grained for
  a sequence-level EBM), use the EBM energy to guide a diffusion process over full sequences. The
  score matching framework in this paper could replace the token-masking approach that failed in DoT.
- **Concrete experiment:** Redesign DoT using EBM-diffusion formulation: energy over full sequence
  guides noise schedule (high energy = high noise = mask more tokens). Compare vs failed per-token
  masking from exp1171.
- **When to incorporate:** Milestone .92 DoT diagnosis task, exp1186.

### Active Inference AI Systems for Scientific Discovery
- **Paper:** arXiv 2506.21329 (2026).
- **What:** Integrates active inference (Friston free-energy framework) with LLMs using chain-of-thought
  reasoning and Expected Free Energy (EFE) minimization. Demonstrates EFE as a principled decision
  criterion for agentic AI systems in scientific discovery tasks.
- **Relevance to Carnot:** Phase 4 (exp1165) implemented Blocked Gibbs free-energy minimization for
  ARC-AGI-3. This paper provides independent theoretical grounding: EFE minimization IS what Carnot's
  Phase 4 implements. The paper's formalization of EFE = Σ_k E_k(z) for multi-verifier systems maps
  directly to Carnot's k=N AND-composed energy F(z) = Σ_k w_k E_k(z). Useful for extending Phase 4
  theory in the position paper and for motivating exp1189 (stronger baseline pilot).
- **Concrete experiment:** Cite in paper Phase 4 section (already in exp1167's scope). No separate
  experiment needed; use for theoretical grounding in the position paper v5.
- **When to incorporate:** Paper integrity task (exp1180/1181) — add citation to Section 7.

### Physical Analog KANs via Reconfigurable Nonlinear Processors
- **Paper:** arXiv 2602.07518 (February 2026).
- **What:** Implements KANs using physical analog hardware (reconfigurable nonlinear processors).
  Demonstrates that KAN spline activations map naturally to analog nonlinear hardware, providing
  inference without digital multiplication — all operations are physical analog transformations.
- **Relevance to Carnot:** BiKA (arXiv 2602.23455) achieves multiply-free KAN via bit-shifts. Physical
  analog KANs go further: no digital computation at all. The SOS-KAN verifier (AUROC=0.9902) could
  potentially be implemented as an analog circuit, enabling sub-nanosecond energy evaluation on future
  Extropic-class hardware. Medium-term: after Extropic Z1 ships, this architecture is the hardware
  target for the verifier.
- **Concrete experiment:** Not yet — monitor Extropic Z1 availability. File for hardware roadmap.
- **When to incorporate:** .93+ after Extropic Z1 early-access or comparable analog hardware.

## 2026-05-02 Scan (Milestone 2026.04.91 Planning)

### TinyV: Reducing False Negatives in Verification Improves RL Training
- **Paper:** arXiv 2505.14625 (May 2025).
- **Source:** https://arxiv.org/abs/2505.14625
- **What:** Reveals 38% false negative rate in standard LLM output verifiers — correct responses
  are rejected due to format/notation mismatches. TinyV replaces strict exact-match with a
  lightweight model that tolerates superficial variation. Directly improves RL training signal
  quality by reducing reward noise from false negatives.
- **Relevance to Carnot:** GRPO v4 (exp1159) achieved +10% improvement with structural warm-up;
  false negatives in the ThinkPRM/energy verifier reward signal may still be introducing noise.
  TinyV's false-negative correction logic applied to Carnot's PRM threshold could improve GRPO v5.
  Also relevant for calibration: the 38% FNR means Carnot's SECL calibration fix (exp1157) may
  still leave 38% of real improvements uncaptured in the reward signal.
- **Concrete experiment:** Apply TinyV-style false-negative correction to GRPO v5 energy reward:
  when verifier confidence is below threshold, abstain from reward signal rather than penalizing.
  Compare action_count and improvement_over_baseline vs GRPO v4 baseline (+10%).
- **When to incorporate:** Milestone .91 GRPO v5 design.

### BiKA: Ultra-Lightweight Multiply-Free KAN Hardware Accelerator
- **Paper:** arXiv 2602.23455 (February 2026).
- **Source:** https://arxiv.org/abs/2602.23455
- **What:** Proposes a multiply-free KAN architecture where spline activations are approximated
  via bit-shifts and additions only. Reports 27.73-51.54% reduction in hardware resource usage
  vs standard KAN implementations while maintaining competitive accuracy.
- **Relevance to Carnot:** SOS-KAN (SOSKANEnergyV3, AUROC=0.9902) is the cheapest load-bearing
  verifier; MetaCluster compressed it to 5.03x smaller (exp1148). BiKA is the next step toward
  NPU-native deployment: replace multiplications with bit-shift approximations in the spline layers.
  This enables the verifier to run on AMD XDNA NPU and future Extropic Z1 without floating-point
  hardware. Pairs with KANELE (arXiv 2512.12850) LUT-based design flow.
- **Concrete experiment:** Apply BiKA multiply-free architecture to SOSKANEnergyV3; compute
  RM/BOP/NABS metrics from arXiv 2604.03345; compare hardware cost vs standard SOS-KAN, MetaCluster
  compressed variant, and KANELE blueprint. Report NPU deployment feasibility.
- **When to incorporate:** Milestone .91 hardware portfolio analysis.

### GRPO is Secretly a Process Reward Model
- **Paper:** arXiv 2509.21154 (September 2025).
- **Source:** https://arxiv.org/abs/2509.21154
- **What:** Proves that GRPO with an outcome reward model is mathematically equivalent to a
  PRM-aware RL objective. The group-normalized advantage in GRPO implicitly computes
  step-level credit assignment. Clarifies when outcome-level vs step-level reward gives different
  gradients.
- **Relevance to Carnot:** Carnot's GRPO uses ThinkPRM v2 (AUROC=0.9946) as the energy-reward
  signal. This paper confirms the theoretical grounding: GRPO + ThinkPRM is PRM-aware RL by
  construction. Useful when explaining the self-learning mechanism in the position paper v4.
- **Concrete experiment:** Use this paper as theoretical justification for the GRPO+PRM framing
  in the position paper v4 Phase 4 section. No separate experiment needed; cite in paper.
- **When to incorporate:** Paper v4 Phase 4 integration (milestone .91 exp1167).

### Corrective Diffusion Language Models
- **Paper:** arXiv 2512.15596 (December 2025).
- **Source:** https://arxiv.org/abs/2512.15596
- **What:** Post-training method that teaches diffusion language models to correct targeted
  tokens via masked diffusion and parallel iterative refinement. Achieves correction-oriented
  text generation without full regeneration.
- **Relevance to Carnot:** The Diffusion of Thought (DoT) inference mode proposed in
  known-issues.md needs a concrete implementation pattern. Corrective diffusion provides
  a masked-token refinement scheme where Carnot's energy gradient identifies WHICH tokens
  to remask, and the diffusion model proposes corrections. This is more targeted than
  full DoT and better matches Carnot's verifier-guided repair philosophy.
- **Concrete experiment:** Implement DoT inference mode using corrective diffusion's masked
  refinement pattern: (1) energy gradient identifies high-violation tokens → mark as remask
  candidates; (2) diffusion step proposes corrections for marked tokens; (3) re-verify.
  Compare vs full Blocked Gibbs and vs autoregressive repair on FoVer + 50 GSM8K.
- **When to incorporate:** Milestone .91 Diffusion of Thought inference mode.

### ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence
- **Paper:** arXiv 2603.24621 (March 2026).
- **Source:** https://arxiv.org/abs/2603.24621
- **What:** Third iteration of the ARC-AGI benchmark, requiring goal inference and sequential
  planning over novel grid transformations. Frontier autoregressive LLMs all score below 1%;
  Seed IQ (active inference) scores 100% with 115% human action-efficiency.
- **Relevance to Carnot:** Phase 4 arc: Carnot's Blocked Gibbs free-energy minimization on
  k=5 AND-composed verifier = active inference under a different name. The exp1154 snap operator
  confirmed ≥95% valid action coverage. Pilot on 10 representative ARC-AGI-3 puzzle types
  is the empirical gate for lifting the arXiv publication hold.
- **Concrete experiment:** Phase 4 ARC-AGI-3 minimal pilot: 10 synthetic puzzles, Blocked Gibbs
  free-energy minimization → action selection → compare action-count vs greedy baseline and
  Seed IQ published numbers (VC33: 173 actions, FT09: 75 actions, LS20: 433 actions).
- **When to incorporate:** Milestone .91 Phase 0 CRITICAL (exp1165).

## 2026-05-02 Scan (Milestone 2026.04.90 Planning)

### KANELÉ: KAN FPGA Deployment with 2700x Speedup
- **Paper:** arXiv 2512.12850 (December 2025).
- **Source:** https://arxiv.org/abs/2512.12850
- **What:** First systematic design flow for implementing KANs on FPGAs via co-optimization
  of training, quantization, and pruning. Spline activations map naturally to LUT structures.
  Reports up to 2700x speedup and orders-of-magnitude resource savings over prior KAN-on-FPGA
  approaches, with competitive inference quality.
- **Relevance to Carnot:** SOS-KAN (SOSKANEnergyV3, AUROC=0.9902) is the load-bearing cheap
  verifier. MetaCluster compression (.89 exp1148) achieved 5.03x size reduction. KANELÉ is the
  next step: a literal FPGA blueprint that turns the compressed KAN into an RTL-synthesizable
  LUT table, targeting the KV260 or a future Alveo for sub-microsecond energy evaluation.
- **Concrete experiment:** Generate a KANELÉ-style FPGA LUT blueprint from the exp1148
  compressed SOSKANEnergyV3 checkpoint; compute hardware complexity metrics (RM, BOP, NABS)
  per arXiv 2604.03345; compare estimated FPGA latency vs CPU baseline (289ms).
- **When to incorporate:** Milestone .90 hardware phase.

### Hardware-Oriented Inference Complexity of KANs
- **Paper:** arXiv 2604.03345 (April 2026).
- **Source:** https://arxiv.org/abs/2604.03345
- **What:** Derives platform-independent metrics for KAN hardware complexity: Real Multiplications
  (RM), Bit Operations (BOP), and Number of Additions and Bit-Shifts (NABS), across B-spline,
  Gaussian RBF, Chebyshev, and Fourier KAN variants. Computable from network structure alone,
  enabling early-stage architectural comparison without full synthesis.
- **Relevance to Carnot:** Enables objective comparison between SOS-KAN, compressed SOS-KAN
  (exp1148), and the KANELÉ FPGA target. Use RM/BOP/NABS to guide model selection for NPU/FPGA
  deployment without waiting for synthesis runs.
- **Concrete experiment:** Include in the KANELÉ blueprint experiment (.90) as the complexity
  analysis output. Compare original SOS-KAN vs MetaCluster-compressed vs FPGA-LUT form.
- **When to incorporate:** Milestone .90 alongside KANELÉ blueprint.

### SECL: Self-Calibrating Language Models via Test-Time Discriminative Distillation
- **Paper:** arXiv 2604.09624 (April 2026).
- **Source:** https://arxiv.org/abs/2604.09624
- **What:** Test-time training method that exploits the gap between LLMs' expressed confidence
  and their better-calibrated internal signal ("Is this correct?" token probability). Only
  adapts when input distribution shifts (6-26% of the question stream). Reduces Expected
  Calibration Error by 56-78% across models and domains.
- **Relevance to Carnot:** exp1145 (.89) achieved 91.7% TP on Goodfire exemplars but
  false_positive_rate=0.96 — the threshold adjustment was too aggressive. SECL's discriminative
  distillation provides a theoretically grounded recalibration: train the cheap-tier router on
  the discriminative "Is this response correct?" signal rather than a fixed threshold, removing
  the TP/FP trade-off. This directly addresses the .89 bottleneck identified in the retro.
- **Concrete experiment:** Replace exp1145's fixed threshold adjustment with SECL-style
  test-time discriminative calibration on the Goodfire exemplar corpus; target TP>=80%, FPR<=0.30.
- **When to incorporate:** Milestone .90 verifier calibration phase.

### Graph-GRPO: Structural Constraint Reinforcement Learning
- **Paper:** arXiv 2603.10395 (March 2026).
- **Source:** https://arxiv.org/abs/2603.10395
- **What:** Two-stage GRPO training that first optimizes with a simplified structural constraint
  objective, then uses the resulting checkpoint to initialize full RL with multi-objective
  optimization. Designed for molecule generation but the structural-constraint-first pre-training
  idea is domain-general.
- **Relevance to Carnot:** exp1146 GRPO reflection reward v3 produced +2.86pp vs +8.51pp from
  exp1129. The structural-constraint-first phase could explain the gap: GRPO v3 mixed the
  reflection reward (structural) with ThinkPRM (semantic) from the start. A Graph-GRPO-inspired
  warm-up phase that trains ONLY on reflection reward r_reflect for the first N steps, then
  mixes in ThinkPRM, could improve stability.
- **Concrete experiment:** Incorporate structural-first GRPO warm-up into exp1159 GRPO v4.
- **When to incorporate:** Milestone .90 GRPO v4 design.

### Active Inference for Self-Organizing Multi-LLM Systems
- **Paper:** arXiv 2412.10425 (December 2024).
- **Source:** https://arxiv.org/abs/2412.10425
- **What:** Integrates active inference with LLM agents as a cognitive layer dynamically
  adjusting prompts and search strategies through principled information-seeking behavior.
  The active inference framework computes expected free energy to select information-gathering
  actions that minimize uncertainty about the world model.
- **Relevance to Carnot:** Phase 4 track (committed, mandatory for .90+): Carnot's k=5
  AND-composed verifier ensemble serves as the calibrated free-energy approximation while the
  LLM substrate retains autoregressive infrastructure. This paper provides a concrete multi-LLM
  instantiation pattern that maps directly to Carnot's existing verifier + LLM architecture.
- **Concrete experiment:** Phase 4 active inference pilot: implement a minimal variational
  free-energy loop using exp1128 k=5 ensemble as the precision-weighted prediction error;
  validate on 50 FoVer examples (Phase-4 committed track).
- **When to incorporate:** Milestone .90 Phase 4 mandatory track.

## 2026-05-02 Supplemental Scan (Milestone 2026.04.89 Planning)

### NRGPT: An Energy-based Alternative for GPT
- **Paper:** arXiv 2512.16762 (December 2025); ICLR 2026 paper page also visible on
  OpenReview.
- **Source:** https://arxiv.org/abs/2512.16762 and
  https://openreview.net/pdf?id=B3Muyi2zgo
- **What:** Recasts a GPT-style causal language model as an energy-based dynamical
  system where the inference update explores token states on an energy landscape. The
  authors report competitive small-language-model results on Shakespeare, ListOPS, and
  OpenWebText, with evidence that the energy landscape regularizes overfitting.
- **Relevance to Carnot:** Strengthens the Phase 3 bridge: Carnot does not have to
  leap directly from verifier wrapper to full EBT. A minimal GPT-to-energy rewrite can
  be studied as an intermediate "energy-native autoregressive" baseline using FoVer
  traces and local GGUF generations.
- **Concrete experiment:** Add a small NRGPT-inspired recurrent energy block to the
  Phase 3 continuous EBM prototype and compare ListOPS/FoVer trace energy ordering
  against the existing DBAE/continuous-EBM baseline.
- **When to incorporate:** Milestone .90+ after .89 closes the arXiv and certificate
  blockers; use as a Phase 3 prototype seed, not a .89 critical-path task.

### Transformers as Intrinsic Optimizers: Forward Inference through the Energy Principle
- **Paper:** arXiv 2511.00907 (November 2025).
- **Source:** https://arxiv.org/abs/2511.00907
- **What:** Presents a unified energy view of transformer attention: standard softmax
  attention can be interpreted as Helmholtz-free-energy minimization, and variants based
  on momentum, Nesterov acceleration, and Newton-style updates can be derived from the
  same framework.
- **Relevance to Carnot:** EBT-Policy already motivates adaptive Langevin/dynamic-stop
  inference; this paper gives a transformer-internal analogue. Carnot's Phase 3
  continuous-latent prototype should test whether Nesterov-style energy updates reduce
  the number of verifier-gradient refinement steps.
- **Concrete experiment:** In a future continuous-EBM prototype, compare plain
  gradient descent, momentum, Nesterov, and approximate-Newton latent repair on FoVer
  trace embeddings. Metrics: energy decrease per step, invalid-output residual, and
  alpha_t sensitivity.
- **When to incorporate:** Phase 3 prototype follow-up after .89's bounded-verification
  work.

### DiffuTruth / The Energy of Falsehood
- **Paper:** arXiv 2602.11364 (February 2026).
- **Source:** https://arxiv.org/abs/2602.11364 and
  https://huggingface.co/papers/2602.11364
- **What:** Uses a discrete text diffusion reconstruction stress test and an NLI critic
  to define a semantic energy for factual claims. Reports unsupervised FEVER AUROC 0.725
  and improved zero-shot generalization on HOVER.
- **Relevance to Carnot:** This is a concrete thermodynamic factuality probe that is
  orthogonal to k=5 math/code verifiers and HalluGuard routing. It is most useful for
  TruthfulQA/HaluEval-style factual grounding, where Carnot's current strong signals are
  weaker than in arithmetic/code.
- **Concrete experiment:** Add a DiffuTruth-style "semantic reconstruction stress" probe
  for factual claim exemplars and measure whether it raises Goodfire cheap-tier TP rate
  without increasing false positives.
- **When to incorporate:** Milestone .90 factual-grounding phase, or .89 only if the
  Goodfire cheap-tier distillation task needs another feature source.

### EBT Metacognitive Reasoning for Code Generation
- **Paper:** OpenReview MetaGenAI 2025 poster, "A Pipeline for Assessing Metacognitive
  Reasoning in Energy-Based Transformers while Generating Code" (published November
  2025).
- **Source:** https://openreview.net/forum?id=FrY7CU3U3p
- **What:** Proposes feedback-aware EBT inference for code generation, dynamically
  scaling the number of forward passes based on external feedback and optional
  human-in-the-loop control.
- **Relevance to Carnot:** Carnot's HumanEval win came from verification/repair outside
  the generator. This paper suggests a benchmark shape for Phase 3: the energy model
  should decide when more "thinking depth" is warranted during code repair, not use a
  fixed iteration count.
- **Concrete experiment:** Extend the HumanEval repair harness with an adaptive
  energy-depth controller: stop when verifier energy delta plateaus; allocate more
  repair iterations only when AST/Z3 residuals remain high.
- **When to incorporate:** Milestone .90 code-verification scaling, after .89 CCTU
  adapter broadens agentic benchmark coverage.

### MCP Solver: Symbolic Solvers via Model Context Protocol
- **Repo / paper:** `szeider/mcp-solver`; companion SAT 2025 paper "Bridging Language
  Models and Symbolic Solvers via the Model Context Protocol."
- **Source:** https://github.com/szeider/mcp-solver
- **What:** MCP server exposing MiniZinc, PySAT, MaxSAT, Z3, and Clingo tools to LLMs
  through a common edit/solve interface.
- **Relevance to Carnot:** CCTU and tool-use verification need executable validators
  and symbolic backends. Carnot already has an MCP server; this repo offers a direct
  integration pattern for solver-backed tool-use constraints without inventing another
  protocol.
- **Concrete experiment:** In the CCTU adapter, compare Carnot's existing Z3 verifier
  path to an MCP-solver-style backend shim for MiniZinc/PySAT tasks.
- **When to incorporate:** Milestone .89 exp1144 implementation detail or .90
  tool-use broadening.

### PyCSP3 Models Repository
- **Repo:** `xcsp3team/PyCSP3-models`
- **Source:** https://github.com/xcsp3team/PyCSP3-models
- **What:** MIT-licensed repository with 400+ CSP/COP models and data, categorized
  into academic, crafted, realistic, recreational, and single-instance problem families.
- **Relevance to Carnot:** Provides a broad, maintained source of constraint problems
  for WOPR cartridge hardness audits and RandCSPBench-style easy-instance avoidance.
  It is especially useful for adding non-puzzle industrial/academic CSPs to the Ising
  verifier benchmark suite.
- **Concrete experiment:** Add a PyCSP3 import/translation audit to the WOPR/CSP harness:
  select 10 small CSPs, translate to Carnot energy terms, and compare against a classical
  solver baseline.
- **When to incorporate:** Milestone .90+ after Slitherlink rescue; use as benchmark
  expansion rather than blocking .89.

## 2026-05-02 arxiv/OpenReview Scan (Milestone 2026.04.89 Planning)

### BEAVER: An Efficient Deterministic LLM Verifier
- **Paper:** arXiv 2512.05439 (December 2025); ICLR 2026 VerifAI-2 workshop.
- **Source:** https://arxiv.org/abs/2512.05439 and https://huggingface.co/papers/2512.05439
- **What:** Computes deterministic, sound probability bounds that an LLM's output distribution
  satisfies prefix-closed semantic constraints. Uses token-trie/frontier data structures and
  reports 6-8x tighter bounds plus 3-4x more high-risk-instance discovery than baselines.
- **Relevance to Carnot:** Carnot currently certifies sampled outputs, not the probability mass
  of all possible bad outputs. BEAVER is the clearest path to adding a certificate tier above
  k=5 AND-compose: "given this prompt and verifier, at most p_bad mass violates constraints."
- **Concrete experiment:** Build a BEAVER-lite bounder for arithmetic prefix constraints on
  SOTA local GGUF models, compare bound tightness and runtime against empirical sampling.
- **When to incorporate:** Milestone .89 Phase 1 or 2, after .88 fixed k=5 AUROC.

### HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs
- **Paper:** arXiv 2601.18753 (January 2026, revised March 2026); accepted ICLR 2026.
- **Source:** https://arxiv.org/abs/2601.18753 and https://huggingface.co/papers/2601.18753
- **What:** Decomposes hallucination risk into data-driven mismatch and reasoning-driven
  decoding instability. Introduces an NTK-geometry score evaluated across 10 benchmarks,
  11 baselines, and 9 LLM backbones.
- **Relevance to Carnot:** .88 Goodfire results showed Tier 3 k=5 catches all curated
  exemplars, while early learned tiers remain weak. HalluGuard's two-source decomposition
  can become a routing feature: send representation-mismatch cases to factual/semantic
  checks and decoding-instability cases to energy-guided repair.
- **Concrete experiment:** Add HalluGuard-style NTK features to the cascade router and test
  whether Goodfire mixed verdicts become explainable by data-vs-reasoning failure class.
- **When to incorporate:** Milestone .89 cascade calibration.

### CCTU: A Benchmark for Tool Use under Complex Constraints
- **Paper:** arXiv 2603.15309 (March 2026).
- **Source:** https://arxiv.org/abs/2603.15309 and https://huggingface.co/papers/2603.15309
- **What:** 200 tool-use tasks with explicit complex constraints, averaging seven constraint
  types and >4,700-token prompts. Includes executable step-level constraint validation.
  No tested model exceeded 20% completion when all constraints had to be satisfied.
- **Relevance to Carnot:** This is a natural FR-12 benchmark for agentic verification. It
  stresses resource, behavior, toolset, and response constraints, not just GSM8K arithmetic.
- **Concrete experiment:** Build a 25-task CCTU micro-benchmark adapter that runs local SOTA
  GGUF models through Carnot's verifier cascade and executable validator.
- **When to incorporate:** Milestone .89 or .90; strong candidate for the next non-GSM8K
  credibility benchmark.

### Benchmarking GNNs in Solving Hard Constraint Satisfaction Problems / RandCSPBench
- **Paper:** arXiv 2602.18419 (February 2026, revised March 2026).
- **Source:** https://arxiv.org/abs/2602.18419
- **What:** Physics-grounded benchmark suite for random CSPs in the hard phase-transition
  regime. Finds classical methods still outperform current GNN approaches on genuinely
  hard CSPs.
- **Relevance to Carnot:** Prevents another "easy-instance win" trap. WOPR cartridges and
  verifier benchmarks should include hard-regime CSP instances, not just human-solvable toys.
- **Concrete experiment:** Add RandCSPBench-style hard instances to the WOPR/game-cartridge
  harness and compare Carnot Ising/KAN against WalkSAT/Belief Propagation baselines.
- **When to incorporate:** Milestone .89 WOPR rescue/hardness audit.

### HardNet++: Nonlinear Constraint Enforcement in Neural Networks
- **Paper:** arXiv 2604.19669 (April 2026).
- **Source:** https://arxiv.org/abs/2604.19669
- **What:** Differentiable constraint layer that iteratively adjusts neural outputs through
  damped local linearizations to satisfy linear and nonlinear equality/inequality constraints.
- **Relevance to Carnot:** Directly relevant to verifier repair: instead of only scoring and
  asking an LLM to repair, continuous numeric/certified domains can project outputs back into
  the feasible set before returning to text.
- **Concrete experiment:** Prototype a HardNet++-style projection repair layer for continuous
  arithmetic/range constraints and compare it to current prompt-based repair.
- **When to incorporate:** Milestone .89 Phase 3 repair-quality work.

### KKT-Hardnet: Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints
- **Paper:** arXiv 2507.08124 (July 2025; revised August 2025).
- **Source:** https://arxiv.org/abs/2507.08124 and https://github.com/SOULS-TAMU/kkt-hardnet
- **What:** Enforces constraints to machine precision by differentiably projecting outputs onto
  a feasible region through KKT conditions, avoiding soft penalty balancing.
- **Relevance to Carnot:** Complements HardNet++ with a KKT projection route. Useful for
  Phase 3 continuous latent repair where the energy minimizer must obey hard constraints by
  construction.
- **Concrete experiment:** Compare KKT-Hardnet vs HardNet++ projection on small Carnot
  constraint-repair tasks, with violation residual as the primary metric.
- **When to incorporate:** Milestone .89 if repair projection is prioritized; otherwise .90.

### EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities
- **Paper:** arXiv 2510.27545 (October 2025).
- **Source:** https://arxiv.org/abs/2510.27545 and https://energy-based-transformers.github.io/related.html
- **What:** Extends Energy-Based Transformers to action-trajectory policies. Reports fewer
  inference steps than diffusion policies on some tasks, energy-scaled Langevin dynamics,
  dynamic stopping by gradient/energy convergence, and zero-shot recovery behavior.
- **Relevance to Carnot:** Provides a concrete training/stability recipe for Phase 3 continuous
  latent reasoning: energy-scaled step sizes, pre-sample normalization, Nesterov acceleration,
  gradient clipping, and dynamic compute allocation.
- **Concrete experiment:** Apply EBT-Policy's adaptive Langevin/dynamic-stop recipe to the
  Phase 3 continuous EBM prototype on FoVer latent traces; measure convergence steps and
  alpha_t sensitivity.
- **When to incorporate:** Milestone .89 Phase 4 prototype seed.

### MetaCluster: Enabling Deep Compression of Kolmogorov-Arnold Network
- **Paper:** arXiv 2510.19105 (October 2025, revised February 2026).
- **Source:** https://arxiv.org/abs/2510.19105
- **What:** Compresses KAN coefficient vectors with a meta-learner plus centroid codebook,
  reporting up to 80x parameter reduction on standard tasks and 124.1x on equation modeling.
- **Relevance to Carnot:** SOS-KAN/k=5 now works after .88 but KAN memory footprint matters
  for local-first edge deployment and NPU/FPGA portability. MetaCluster could shrink KAN
  verifier checkpoints without losing AUROC.
- **Concrete experiment:** Compress the .88 SOSKANEnergyV3 checkpoint with a centroid codebook
  and compare AUROC, latency, and serialized size.
- **When to incorporate:** Milestone .89 after k=5 checkpoint stabilization.

### Energy-Time-Accuracy Tradeoffs in Thermodynamic Computing
- **Paper:** arXiv 2601.04358 (January 2026).
- **Source:** https://arxiv.org/abs/2601.04358
- **What:** Derives energy-delay-deficiency limits for thermodynamic computing and quasi-optimal
  control protocols for stochastic sampling without prior knowledge of the solution.
- **Relevance to Carnot:** .88 KV260 v4 tuning improved KL but missed the 0.05 gate. This paper
  gives a principled framing for "accuracy vs time vs energy" rather than treating KL alone as
  the only hardware acceptance metric.
- **Concrete experiment:** Add EDDP-style energy-time-accuracy reporting to SamplerBackend
  diagnostics for CPU Gibbs, KV260 simulation, and future TSU/thrml runs.
- **When to incorporate:** Milestone .89 hardware diagnostics.

### Extropic XTR-0 / Z1 Hardware Status
- **Source:** https://extropic.ai/hardware and https://extropic.ai/software
- **What:** XTR-0 is described as a Q3 2025 experimental testing platform with low-latency
  communication between Extropic chips and a traditional processor. Z1 is listed as early
  access 2026, with hundreds of thousands of probabilistic circuits per chip and millions
  per card. THRML remains the public JAX simulation stack for TSU-style PGMs/EBMs.
- **Relevance to Carnot:** User direction already re-scoped KV260 to POC tier. Z1 early access
  is now close enough that Carnot should prepare a concrete hardware-access packet: minimal
  EBM kernels, SamplerBackend API requirements, and acceptance tests.
- **Concrete experiment:** Draft an Extropic early-access integration packet and THRML parity
  benchmark so Carnot can move when Z1/XTR access opens.
- **When to incorporate:** Milestone .89 hardware path.

### Logical Intelligence Kona/Aleph Updates
- **Source:** https://logicalintelligence.com/kona-ebms-energy-based-models,
  https://logicalintelligence.com/blog/aleph-solves-putnambench, and
  https://logicalintelligence.com/blog/energy-based-models-for-reasoning
- **What:** Kona is positioned as a non-autoregressive EBRM for critical systems, with LLMs
  used for interface/orchestration. Aleph is described as an orchestration layer using Lean
  proof checking, reportedly solving 668/672 PutnamBench problems when paired with GPT-5.2.
- **Relevance to Carnot:** Strong external validation of Carnot's "LLM as interface, EBM as
  verifier/reasoner" framing. Also sharpens the competitive benchmark: formal proof or tool-use
  verification, not only GSM8K.
- **Concrete experiment:** Add a Lean/SMT proof-certificate micro-benchmark to Carnot's
  verifier suite, using local SOTA GGUF generation plus deterministic Lean/Z3 checking.
- **When to incorporate:** Milestone .89 or .90, depending on arXiv submission urgency.

### MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination
- **Paper:** arXiv 2603.24579 (March 2026).
- **Source:** https://arxiv.org/abs/2603.24579 and https://github.com/Qwen-Applications/MARCH
- **What:** Uses solver/proposer/checker agents with deliberate information asymmetry, then
  trains the pipeline by multi-agent reinforcement learning to reduce hallucination and
  self-confirmation bias.
- **Relevance to Carnot:** Carnot already separates extraction, verification, and repair. MARCH
  suggests a self-learning variant where claim extraction and checking are trained with blinded
  roles to avoid the generator's original error contaminating the verifier.
- **Concrete experiment:** Create an information-asymmetric claim-check loop over Goodfire/CCTU
  exemplars and compare against current single-pass LLM-as-extractor.
- **When to incorporate:** Milestone .89 self-learning phase.

## 2026-05-02 arxiv Scan (Milestone 2026.04.88 Planning)

### DRA-GRPO: Diverse Reasoning Paths for Mathematical Reasoning
- **Paper:** arXiv 2505.09655 (May 2025)
- **What:** Addresses GRPO mode collapse — when N completions per question are semantically
  redundant, gradient signal degrades. Penalizes cosine-similarity among the group's reasoning
  paths; forces exploration. Achieves consistent MATH benchmark improvements across 7B-70B models
  without extra compute.
- **Relevance to Carnot:** exp1118 hit training_wall_budget_hit=True with only 42/50 questions
  completed; the group's 8 completions per question were likely near-identical (mode collapse).
  DRA-GRPO's diversity penalty is the concrete fix for exp1129 (GRPO full training v2): adding
  semantic diversity to the group-relative advantage prevents gradient washout and allows longer
  training runs on fewer examples.
- **Concrete experiment:** Exp 1129 — GRPO full training v2 with DRA-GRPO diversity penalty;
  n_training=100, budget_s=600, compare advantage_stdev with and without diversity term.
- **When to incorporate:** Milestone .88 exp1129.

### Why Does Self-Distillation (Sometimes) Degrade Reasoning Capability?
- **Paper:** arXiv 2603.24472 (March 2026)
- **What:** Analyzes failure modes of self-distillation for mathematical reasoning. Key finding:
  self-distillation degrades when the teacher and student share a "reasoning mode" — the student
  amplifies existing errors rather than learning from successes. The failure correlates with
  response diversity collapse (all outputs converge to the same reasoning path).
- **Relevance to Carnot:** Theoretical grounding for why 3 consecutive RLVR+SSD attempts
  (exp1083/1099/1110) produced honest negatives. The binary reward (correct/incorrect) drove
  homogenization; GRPO's group-relative advantage avoids this. Confirms Carnot's architectural
  pivot from SSD to GRPO+energy is theoretically well-motivated.
- **When to incorporate:** Background context for exp1129 GRPO v2 design; cite in position paper
  §4 (self-distillation limitations) as additional motivation for energy-grounded GRPO.

### Continuous Energy Ising Machine via Difference-of-Convex Programming
- **Paper:** arXiv 2509.01928 (September 2025)
- **What:** Relaxes binary spins to continuous [-1,+1] variables; decomposes the Ising energy into
  difference-of-convex (DC) components; solves via alternating DC-Prog iterations. Achieves better
  optima than simulated annealing on benchmark Ising instances by escaping binary quantization
  traps during search, then rounding to binary at end.
- **Relevance to Carnot:** This is an alternative continuous-to-Ising search strategy that
  complements our Gray-code visible-spin encoder. The DC relaxation path avoids the period-2
  oscillation problem that exp1094/1122 found in KV260 synchronous updates — the continuous
  solver has no fixed-point instability. Could be the fallback when FPGA Glauber exceeds KL
  threshold. Position paper Phase 2a section should cite.
- **Concrete experiment:** Future — incorporate DC-Prog as a software Ising solver alternative to
  Python Gibbs in the SamplerBackend; benchmark KL vs continuous v4 Python sim.
- **When to incorporate:** Milestone .89+ after KV260 v4 hardware synthesis is unblocked.

### Self-Adaptive Ising Machines for Constrained Optimization via Lagrange Relaxation
- **Paper:** arXiv 2501.04971 (January 2025)
- **What:** Ising machines typically require manual penalty coefficient tuning (λ) for hard
  constraints. This paper derives a self-adaptive scheme where λ updates automatically based on
  constraint violation measurements. Achieves near-oracle constraint satisfaction without hand-
  tuning. Works with FPGA Ising machines and simulated annealing.
- **Relevance to Carnot:** Directly fixes the KV260 v4 parameter sensitivity problem (exp1122:
  KL=0.134 at best alpha=0.1). Instead of sweeping beta/alpha manually (exp1134), the self-
  adaptive scheme learns the optimal penalty coefficient automatically. The Lagrange dual update
  rule is hardware-friendly: just increment λ by violation_count × learning_rate after each sweep.
- **Concrete experiment:** Exp 1134 (KV260 v4 parameter tuning) — add self-adaptive λ update to
  the Python simulation and measure whether KL drops below 0.05 without manual beta sweep.
- **When to incorporate:** Milestone .88 exp1134.

### CPPO: Accelerating Group Relative Policy Optimization
- **Paper:** arXiv 2503.22342 (March 2026)
- **What:** Reuses responses from nearby timesteps as "proxy group members" instead of generating
  N fresh completions per question. Achieves 3.48x wallclock speedup on MATH benchmark with no
  accuracy loss. Proxy group members are selected by embedding similarity to current question.
- **Relevance to Carnot:** exp1118 hit training_wall_budget=240s with 42/50 questions; CPPO's
  proxy group reuse would complete all 100 training questions within the same 600s budget. The
  FoVer corpus (7329 pairs) provides a natural similarity pool for proxy selection. Integration
  cost: ~20 lines of code on top of the existing GRPO loop in exp1118.
- **Concrete experiment:** Exp 1129 (GRPO v2) — use CPPO proxy reuse strategy; target 100
  training questions within 600s budget; compare advantage_stdev with/without proxy reuse.
- **When to incorporate:** Milestone .88 exp1129.

### GRPO with Reflection Reward for Self-Reflective Mathematical Reasoning
- **Paper:** arXiv 2603.14041 (March 2026)
- **What:** Integrates a reflection reward into GRPO that scores whether the model identifies
  and corrects its own errors across multiple attempts. The reflection reward r_reflect = Δ(accuracy)
  between attempt 1 and attempt k, encouraging self-correction. +4.7pp on MATH-500.
- **Relevance to Carnot:** Carnot's verify-repair cycle IS a structured reflection loop: verify
  returns a constraint violation signal; repair is the correction attempt. The reflection reward
  is naturally grounded in Carnot's energy: r_reflect = E_before - E_after for each repair step.
  This is a cleaner formulation than the binary RLVR loss.
- **Concrete experiment:** Follow-up to exp1129 — add reflection reward (E_before - E_after per
  repair step) to GRPO advantage; compare to pure ThinkPRM v2 reward signal.
- **When to incorporate:** Milestone .89 (after GRPO full run in .88 establishes baseline).

### HIVE: Hidden-Evidence Verification for Hallucination Detection
- **Paper:** arXiv 2604.26139 (April 2026)
- **What:** Framing hallucination detection as verification against "hidden evidence" — implicit
  constraints derivable from the question and context without external knowledge bases. Achieves
  state-of-the-art on TruthfulQA and HaluEval benchmarks. Uses a chain-of-verification that
  extracts implicit constraints, checks them, and aggregates into a confidence score.
- **Relevance to Carnot:** HIVE's constraint extraction from implicit context is what Carnot's
  LogicExtractor and NLConstraintExtractor attempt. HIVE's verification pipeline maps onto
  Carnot's Tier 2.5 (SymCodeVerifier) for code and Tier 3 (Ising) for constraint satisfaction.
  HIVE results on TruthfulQA/HaluEval are benchmarks Carnot can target for the position paper.
- **When to incorporate:** Milestone .88/89 position paper §5 (Related Work + Carnot benchmarks
  against HIVE on HaluEval-GSM8K subset).

## 2026-05-01 arxiv Scan (Milestone 2026.04.87 Planning)

### Energy-Based Reward Models for Robust Language Model Alignment (EBRM)
- **Paper:** arXiv 2504.13134 (April 2025, revised August 2025)
- **What:** Root-cause analysis of energy inversion on RL-trained model outputs. Frames reward
  overoptimization as a distribution shift problem: when a policy is trained against a reward
  model (verifier), outputs drift out-of-distribution and the reward model's energy assignments
  become miscalibrated. Proposes EBRM with label-noise-aware training and strategic initialization
  to maintain calibration under RL optimization pressure.
- **Relevance to Carnot:** THIS IS THE CANONICAL EXPLANATION for exp1100/exp1115's energy inversion
  finding (mean_correct_energy=0.689 > mean_incorrect_energy=0.621 on SOTA outputs). The mechanism
  is OOD shift: FoVer was trained on base-model GSM8K solutions; SOTA model outputs (Qwen3.6-35B,
  Gemma4-31B) are heavily RL-optimized and outside the training distribution. EBRM's noise-
  filtering + strategic init is the concrete fix. Exp1120 (energy verifier retrain with SOTA
  corpus extension) should implement the EBRM noise-filtering approach.
- **Concrete experiment:** Exp 1120 — retrain SOS-KAN on FoVer v5 (SOTA model outputs included)
  using label-noise-aware training. Gate: does mean_correct_energy < mean_incorrect_energy
  post-retrain?
- **When to incorporate:** Milestone .87 exp1119/exp1120.

### GRPO is Secretly a Process Reward Model
- **Paper:** arXiv 2509.21154 (September 2025)
- **What:** Proves that vanilla GRPO already implements an implicit PRM via Monte Carlo completions
  over shared prefixes. The "hidden PRM" can be surfaced and enhanced without extra training cost.
  The group-relative advantage estimation is equivalent to a per-step reward model trained on
  completion groups sharing a common prefix.
- **Relevance to Carnot:** Establishes that the GRPO training loop IS a natural integration point
  for Carnot's explicit energy verifier. Carnot's ThinkPRM v2 (AUROC=0.9946) can replace GRPO's
  implicit MC-sampled reward with a calibrated energy-based reward, giving higher signal quality
  while preserving GRPO's training dynamics. Simplifies the RLVR+SSD replacement: no need for
  a separate SSD loop; GRPO + energy reward replaces both RLVR and SSD.
- **Concrete experiment:** Exp 1118 — GRPO with ThinkPRM v2 as explicit PRM reward signal.
- **When to incorporate:** Milestone .87 exp1118.

### Adaptive Test-Time Compute Allocation via Constrained Policy Optimization (Lagrangian)
- **Paper:** arXiv 2604.14853 (April 2026)
- **What:** Frames adaptive compute allocation as a Lagrangian dual problem. The global budget
  constraint (total cascade cost across N queries ≤ B) decomposes to per-instance supervised
  classification via the KKT dual variable. A lightweight MLP trained on (input features) →
  (cascade depth) prediction achieves +12.8% on MATH under matched compute budget vs fixed
  allocation. Input features: question length, initial model confidence, CoT step count.
- **Relevance to Carnot:** Directly implements the compute-optimal cascade routing that
  Carnot's Meta-EBM Cascade Router theorized but left as an "exact-DP solution." The Lagrangian
  dual reduction is simpler than the DP formulation and can use Carnot's initial energy score
  as the primary input feature. Exp1100 found SOTA outputs need mean cascade depth 2.20 vs
  FoVer's 1.08 — adaptive routing would right-size depth per query.
- **Concrete experiment:** Exp 1123 — implement Lagrangian cascade router using Carnot energy
  score + ThinkPRM confidence as features; train on FoVer; measure savings vs fixed k=5.
- **When to incorporate:** Milestone .87 exp1123.

### Energy-Efficient p-Bit Simulated Annealer with Dual BRAM Architecture
- **Paper:** arXiv 2602.16143 (February 2026)
- **What:** 800-node fully-connected Ising machine on Xilinx ZC706 (same Zynq-7000 family as
  KV260). Dual BRAM delay-line architecture achieves 50% power reduction, 90% LUT reduction
  vs prior p-bit designs. Update cadence: 50 MHz clock, 800 spins × 16-bit couplings in BRAM.
  Still requires Vivado for synthesis (Zynq-class).
- **Relevance to Carnot:** Demonstrates that the KV260's BRAM architecture is sufficient for
  800+ node Ising machines — 6× above Carnot's current 64-node POC. The 90% LUT reduction
  suggests Carnot's v4 sparse K=16 architecture is over-engineered at N=64; could scale
  directly to N=256 or N=512 with the dual-BRAM pattern.
- **Concrete experiment:** Incorporate dual-BRAM design pattern into v4/v5 Verilog spec review.
- **When to incorporate:** Milestone .87 exp1122 (KV260 v4 Python sim + architecture review).

### A Fully Parallel Probabilistic Ising Machine with Inertia for Real-Time Applications
- **Paper:** arXiv 2604.17109 (April 2026)
- **What:** Adds per-spin inertia (EMA momentum) to p-bit update dynamics to enable fully
  parallel (synchronous) updates without the solution-quality degradation that normally afflicts
  synchronous schemes (period-2 oscillations, detailed-balance violation). Inertia damps
  frustrated spin states, preventing oscillation. 35-150x speedup on 200-spin benchmarks.
  Demonstrated for 5G MIMO detection in real-time (10 MHz decode rate).
- **Relevance to Carnot:** Directly validates Carnot's v4 design choice (inertia + synchronous).
  The paper's proof that inertia enables correct parallel updates is the theoretical foundation
  for exp1109's finding that v3 (sequential, KL=0.025) is correct and v1 (parallel, KL=3.07)
  is not — the inertia term bridges them. The v4 Verilog (ising_sampler_v4.v) already
  implements this; exp1122 should validate it in Python simulation.
- **Concrete experiment:** Exp 1122 — Python simulate v4 inertia dynamics; measure KL(v4||Gibbs).
- **When to incorporate:** Milestone .87 exp1122.

### Reward Under Attack: Analyzing the Robustness and Hackability of Process Reward Models
- **Paper:** arXiv 2603.06621 (February 2026)
- **What:** Three-tier diagnostic framework for PRM hackability: stylistic attacks, logically
  corrupted reasoning, gradient-based adversarial attacks. Key finding: 43% of reward gain
  attributable to stylistic shortcuts (padding, formatting) rather than reasoning quality.
  Near-perfect PRM scores (>0.9) achievable with <4% ground-truth accuracy. Released
  PRM-BiasBench adversarial evaluation suite.
- **Relevance to Carnot:** Confirms Carnot's null-space mimicry attack analysis (memory:
  project_null_space_mimicry_attack.md). The 43% stylistic-shortcut finding is the PRM
  equivalent of Boolean E perfectly mimicking orthogonality stall. PRM-BiasBench is a ready-
  made adversarial test suite to run against Carnot's k=5 AND-composition verifier to measure
  how many stylistic attack exemplars the ensemble catches vs individual verifiers.
- **When to incorporate:** Future milestone — validate k=5 AND-compose robustness on PRM-BiasBench.

### Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling
- **Paper:** arXiv 2505.11730 (May 2025)
- **What:** Derives that step-level verification is compute-optimal only when per-step error
  probability exceeds a threshold; below that threshold, response-level verification dominates.
  The crossover depends on chain length and step error rate. Provides a closed-form criterion.
- **Relevance to Carnot:** Informs when to use ThinkPRM v2 (step-level, expensive) vs SOS-KAN
  (response-level, fast). The criterion can be operationalized using Carnot's per-tier latencies:
  Tier 0a ThinkPRM at ~50ms/query vs Tier 0c SemEnergy at 0.017ms/query. Suggests that SemEnergy
  should be the default path except when step error rate is high.
- **When to incorporate:** Cascade depth optimization (exp1123) and Tier 0a/0c threshold tuning.

## 2026-05-01 arxiv Scan (Milestone 2026.04.86 Planning)

### A Unified Performance–Cost Landscape of Parallel p-bit Ising Machines Based on Update Dynamics
- **Paper:** arXiv 2604.01564 (March 2026)
- **What:** Systematic analysis of synchronous vs asynchronous update schemes in p-bit Ising
  machines under realistic hardware constraints (finite delay, time-multiplexed p-bit reuse,
  limited DAC precision). Key finding: synchronous updates are NOT inherently unstable but
  suffer oscillations under excessive simultaneity. Time-multiplexed p-bit reuse achieves
  stable synchronous operation at less than half the hardware cost of optimized async designs.
  Low-resolution DACs (3-4 bits) suffice when annealing timing is properly tuned.
- **Relevance to Carnot:** DIRECTLY addresses Phase 2a finding (exp1094): KV260 synchronous
  parallel Glauber loses detailed balance on frustrated J (KL=3.07). The paper's time-
  multiplexed p-bit reuse is the design pattern for KV260 v3 bitstream. Exp1109 (KV260
  sequential/time-multiplexed redesign) should use this paper as the implementation reference.
- **Concrete experiment:** Exp 1109 (KV260 Ising sampler v3 — time-multiplexed synchronous
  updates) — target KL(FPGA||Gibbs) < 0.05.
- **When to incorporate:** Milestone .86 Phase 3 FPGA redesign.

### Finite-Time Observability of Oscillatory Instabilities in Synchronous p-bit Dynamics
- **Paper:** arXiv 2603.25910 (March 2026)
- **What:** Synchronous tick-random p-bit dynamics can induce period-2 oscillations on
  frustrated topologies (e.g., antiferromagnetic rings), degrading optimization performance.
  Derives a graph-dependent criterion (finite-time observability matrix) that predicts
  whether unstable modes amplify within a finite observation window — enabling pre-synthesis
  detection of whether a given coupling topology will oscillate.
- **Relevance to Carnot:** Theoretical explanation for exp1094's KL=3.07 result. The KV260's
  64-spin antiferromagnetic ring (frustrated J) is exactly the topology where synchronous
  oscillations are predicted. The graph criterion can be used to validate whether the
  KV260 v3 bitstream's sequential update scheme avoids the oscillation. Position paper
  Phase 2a finding section should cite both this paper and arXiv 2604.01564.
- **Concrete experiment:** Exp 1109 — verify the redesigned coupling topology satisfies the
  finite-time observability criterion before hardware deployment.
- **When to incorporate:** Milestone .86 FPGA redesign (exp1109) + position paper v3.

### Adversarial Training for Process Reward Models (APRM)
- **Paper:** arXiv 2511.22888 (November 2025)
- **What:** Introduces APRMs where a Generator learns to produce reasoning errors to deceive
  a PRM while the PRM concurrently learns to detect them. +3.4pp over strongest PRM baseline,
  +5.3pp on out-of-distribution tasks. Addresses the core PRM failure mode: poor generalization
  to novel errors due to static training data.
- **Relevance to Carnot:** Phase 1a adversarial verifier robustness audit (exp1106). APRM's
  adversarial generator + verifier co-training is exactly the attack model for measuring
  false-pass rate. The Generator plays the role of the attacker; Carnot's verifier (SOS-KAN,
  ThinkPRM) plays the Discriminator. Use APRM attack patterns to generate the 100-example
  adversarial corpus for exp1106. +5.3pp OOD result also motivates verifier adversarial
  co-training as a future enhancement.
- **Concrete experiment:** Exp 1106 (Phase 1a adversarial audit v2) — use APRM-style attack
  patterns (stylistic padding, formatting, IPT isomorphic perturbation) as the adversarial
  probe. Future: APRM-style co-training to harden Carnot verifiers.
- **When to incorporate:** Milestone .86 Phase 1a (exp1106) + future hardening.

### Reward Hacking in the Era of Large Models: Mechanisms, Emergent Misalignment, Challenges
- **Paper:** arXiv 2604.13602 (April 2026)
- **What:** Survey of reward hacking mechanisms in large model RL. Key finding: optimization
  pressure systematically drives the policy into the null space of the proxy evaluator (the
  "proxy gap"). RLVR creates a proxy gap by rewarding checkable final answers while ignoring
  reasoning steps, incentivizing guessing, fabricated reasoning, and tool misuse. Identifies
  null-space exploitation as the fundamental mechanism.
- **Relevance to Carnot:** Strongest independent confirmation of Carnot's null-space defense
  framework (Phase 1c, Phase 3). The "proxy gap" framing maps directly to Carnot's verifier
  null space. Position paper Section 3 should cite this survey alongside arXiv 2603.06621
  and arXiv 2604.15149 to establish the attack taxonomy. The paper's mitigation analysis
  (diverse verifiers, process-level verification) validates Carnot's architectural approach.
- **Concrete experiment:** No new experiment needed — incorporate into position paper v3
  Section 3 (Related Work + Threat Model).
- **When to incorporate:** Position paper Section 3 + exp1106 context.

### Reinforcement Learning via Self-Distillation (SDPO)
- **Paper:** arXiv 2601.20802 (January 2026)
- **What:** SDPO: policy gradient where advantages are estimated using a self-teacher
  (same model in inference-time mode as the "privileged" teacher). Implementation requires
  minor changes to standard RLVR pipelines. Addresses distribution shift in off-policy
  distillation by keeping teacher and student on-policy.
- **Relevance to Carnot:** Third SSD variant (alongside arXiv 2604.03128 and arXiv 2601.18734)
  for the RLVR+SSD integration track. The self-teacher advantage estimation is compatible
  with Carnot's energy verifier as the reward signal: the teacher generates multiple
  completions, Carnot selects the lowest-energy one, SDPO trains the student on the advantage.
  Particularly relevant for exp1110 (RLVR+SSD v2 non-degenerate corpus) as an alternative
  training algorithm when on-policy distribution matters.
- **When to incorporate:** Exp 1110 RLVR+SSD v2 — compare SDPO vs standard SSD training.

### When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification
- **Paper:** arXiv 2504.01005 (April 2025)
- **What:** Derives a compute-optimal policy for allocating budget between solving (generating
  candidates) and verification (checking candidates). Key result: verification is most
  valuable when solutions are hard to generate but easy to check. Introduces GenRM framing
  where verification is itself a next-token-prediction task.
- **Relevance to Carnot:** Directly informs Carnot's cascade depth optimization question
  (exp1100 found SOTA outputs need mean_depth=2.20 vs FoVer's 1.08). The compute-optimal
  verification budget allocation provides a principled framework for setting per-tier
  exit thresholds. Position paper Section 5 should cite this for the cascade efficiency
  argument.
- **When to incorporate:** Position paper v3 Section 5 (cascade efficiency).

## 2026-05-01 arxiv Scan (Milestone 2026.04.85 Planning)

### Reward Under Attack: PRM Robustness and Hackability
- **Paper:** arXiv 2603.06621 (March 2026)
- **What:** Shows PRMs are systematically exploitable: gradient-based attacks inflate rewards on
  invalid trajectories by 43% via stylistic shortcuts (long preambles, structured formatting)
  rather than correctness signals. The attack surface is the verifier's learned null space.
- **Relevance to Carnot:** Directly confirms the null-space mimicry attack threat model in
  CLAUDE.md. Carnot's Phase 1a adversarial verifier robustness audit (known-issues.md MANDATORY)
  should use the attack patterns from this paper to measure false-pass rate. The paper's
  Isomorphic Perturbation Testing (IPT) is a cheap adversarial probe compatible with FoVer corpus.
- **Concrete experiment:** Exp 1092 (Phase 1a adversarial audit) — use arXiv 2603.06621 attack
  patterns as the adversarial probe suite.
- **When to incorporate:** Milestone .85 Phase 1a adversarial audit.

### LLMs Gaming Verifiers: RLVR Reward Hacking
- **Paper:** arXiv 2604.15149 (April 2026)
- **What:** RLVR models learn to enumerate instance labels instead of inducing relational rules,
  passing extensional verifiers without learning the underlying structure. Introduces Isomorphic
  Perturbation Testing (IPT) as a verifier-agnostic adversarial probe.
- **Relevance to Carnot:** Directly applicable to Carnot's verifier null-space measurement
  (Phase 1c). IPT is a low-cost adversarial check: take a verified output, apply structure-
  preserving permutation, check if energy changes appropriately. If not, the verifier is
  exploitable. Complements arXiv 2603.06621.
- **Concrete experiment:** Exp 1093 (Phase 1c null-space measurement) and Exp 1092 (Phase 1a
  robustness audit) — use IPT as a structural exploit probe.
- **When to incorporate:** Milestone .85 Phase 1a and 1c experiments.

### Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs
- **Paper:** arXiv 2601.18734 (January 2026)
- **What:** Single LLM acts as both teacher and student: teacher generates verified traces
  (access to oracle), student sees only question, trained via per-token divergence on student's
  own rollouts. Achieves stronger token efficiency than standard RLVR and beats off-policy
  distillation. Demonstrates that on-policy alignment is the key to avoiding distribution shift.
- **Relevance to Carnot:** Third concrete SSD variant complementing Zenil/SSD and arXiv
  2604.03128 (Self-Distilled RLVR). The on-policy aspect is directly relevant to Carnot's
  FR-11 RLVR+SSD integration — Carnot's energy verifier provides the oracle signal for the
  teacher. The per-token divergence training objective maps to Carnot's step-level PRM data
  (data/step_level_prm_training.jsonl, 7349 examples from exp1084).
- **When to incorporate:** Exp 1099 (RLVR+SSD integration v1 in .85).

### Efficient Hardware Architecture for Diffusion-Like EBMs
- **Paper:** arXiv 2510.23972 (October 2025)
- **What:** CMOS transistor RNG implementing Denoising Thermodynamic Models (sequential hardware
  EBMs) at ~10,000x lower energy than GPU equivalents. From Extropic co-authors. Demonstrates
  that hardware-native stochasticity (thermodynamic noise) IS computation — not a bug but a
  feature for Gibbs-like samplers.
- **Relevance to Carnot:** Directly supports Carnot's Phase 2 hardware roadmap. The architecture
  confirms the Extropic Z1 design philosophy: use thermodynamic noise as the sampling primitive
  rather than digital RNGs. For the position paper, this provides an independent Extropic-adjacent
  citation showing that sub-milliwatt hardware EBM inference is achievable.
- **When to incorporate:** Position paper Phase 2 hardware section (Exp 1091). Phase 2a sampler
  correctness audit (Exp 1094) — cite as theoretical baseline for expected hardware behavior.

### Programmable k-local Ising Machines and All-Optical KAN on Photonic Platforms
- **Paper:** arXiv 2508.17440 (August 2025)
- **What:** Unifies k-local Ising optimization AND KAN layers on a single photonic platform
  using spatial light modulators with trainable in-situ physical gradients. The ONLY published
  work physically co-locating Ising machines and KAN computation — exactly Carnot's dual-
  primitive architecture (Ising sampling + KAN energy tiers).
- **Relevance to Carnot:** HIGH PRIORITY for the position paper. This paper is the only published
  work with the same hardware-primitive combination as Carnot. The all-optical KAN could serve
  as Phase 3 hardware for Carnot's KAN energy tier, with Ising sampling running on the same chip.
  Cite in position paper Phase 2/3 hardware section as convergent evidence that the Ising+KAN
  co-design is the right hardware direction.
- **When to incorporate:** Position paper v2 (Exp 1091) — cite in hardware motivation section.
  Hardware wishlist: add photonic KAN+Ising platform as Phase 3 hardware candidate.

### Draft-Conditioned Constrained Decoding for Structured Generation
- **Paper:** arXiv 2603.03305 (March 2026)
- **What:** Decouples semantic planning (unconstrained draft) from structural enforcement
  (constrained decode), reducing the "projection tax" in standard constrained decoding. Semantic
  draft is generated first, then constrained decoding is applied with the draft as a conditioning
  signal — yielding locally valid AND semantically correct outputs. Up to +24pp over standard
  constrained decoding on structured benchmarks.
- **Relevance to Carnot:** Repair pipeline upgrade path. Carnot's repair step (after violation
  detection) currently uses simple constrained generation. DCCD could substantially improve
  repair quality by conditioning the constrained rewrite on the original intent. This is directly
  relevant to the GSM8K extraction fix experiment (Exp 1101) — better repair quality on math
  reasoning.
- **When to incorporate:** Exp 1101 (GSM8K extraction + repair fix in .85). Position paper
  Section 4 (architecture) — cite as repair quality upper bound.

### Robust Optimization for Mitigating Reward Hacking with Correlated Proxies
- **Paper:** arXiv 2604.12086 (April 2026)
- **What:** Formalizes when proxy reward correlation to true reward is sufficient to prevent
  hacking. Derives principal conditions for safe proxy reward use and provides mitigation for
  cases where verifiers share correlated failure modes. The r-correlation framework provides
  a principled metric for verifier diversity.
- **Relevance to Carnot:** Phase 1c verifier joint null-space measurement. The r-correlation
  metric from this paper provides a formal bound on AND-composition effectiveness: if verifiers
  have r-correlation > threshold, AND-composition does NOT shrink the joint null space as
  expected. This is the algebraic-geometry confirmation of the pathological joint null space
  problem from CLAUDE.md.
- **When to incorporate:** Exp 1093 (Phase 1c null-space measurement in .85) — use r-correlation
  as the measurement metric alongside dim(∩_i ker E_i).

## 2026-04-30 arxiv Scan (Milestone 2026.04.84 Planning)

### Energy Outcome Reward Model (EORM): 55M-Parameter EBM Verifier
- **Paper:** arXiv 2505.14999 (May 2025)
- **What:** Proposes EORM, a 55M-parameter energy-based verifier that ranks Chain-of-Thought
  solutions. Uses an explicit energy framework rather than a discriminative classifier.
  Achieves 90.7% on GSM8k with 127x fewer parameters than typical reward models. Key:
  the energy function is learned end-to-end from (question, solution, label) triples.
- **Relevance to Carnot:** Independent empirical validation of energy-based CoT ranking at
  minimal parameter counts. EORM's architecture (55M) maps to Carnot's KAN energy tier
  (target: 8.7x fewer params than Ising). Compare EORM's learned energy against Carnot's
  physics-informed (Ising+KAN) energy on FoVer corpus. The 127x parameter efficiency
  framing is precisely the argument Carnot's position paper makes for energy-based verifiers
  over large reward models.
- **Concrete experiment:** Exp 1080 or follow-on — implement EORM-style training on FoVer
  corpus, compare AUROC vs SOS-KAN v3 (0.9545). If similar, cite as independent confirmation.
- **When to incorporate:** Position paper Section 4 (architecture) — cite as independent
  validation. Research comparison in .84 if time permits.

### Process Reward Models Meet Planning: Scalable Step-Level Supervision
- **Paper:** arXiv 2604.17957 (April 2026)
- **What:** Combines Process Reward Models (step-level verifiers) with planning algorithms
  to generate synthetic step-level supervision data at scale. Uses Monte Carlo Tree Search
  (MCTS) to simulate partial reasoning trajectories and assign process-level labels without
  human annotation. Achieves state-of-the-art PRM accuracy on math benchmarks.
- **Relevance to Carnot:** ThinkPRM (Exp 1033, AUROC 0.9885) is Carnot's step-level verifier.
  Its training data bottleneck is the FoVer corpus size (6548 pairs). This paper's MCTS-based
  data generation pattern could expand ThinkPRM training data to 50k+ step-level labeled
  examples using Carnot's own Ising energy as the scoring signal. The planning data generator
  IS a form of self-distillation that satisfies Zenil's α_t > 0 condition (the energy score
  provides the grounding signal at each step).
- **Concrete experiment:** Exp 1084 in .84 — implement MCTS-based data generation using
  Carnot's cascade as the step scorer, generate 10k+ step-level pairs, retrain ThinkPRM.
  Target: AUROC >= 0.995 (beating current 0.9885 with more data).
- **When to incorporate:** Milestone .84 (Exp 1084).

### Trust but Verify! Survey on Verification Design for Test-Time Scaling
- **Paper:** arXiv 2508.16665 (August 2025)
- **What:** Comprehensive survey of LLM verification methods for test-time scaling (TTS).
  Categorizes verifier architectures (discriminative vs. generative vs. energy-based),
  training paradigms (supervised, RL, contrastive), and scaling strategies (parallel
  sampling vs. sequential refinement). Identifies open problems: verifier generalization
  to unseen domains, computational efficiency, step-level vs. outcome-level tradeoffs.
- **Relevance to Carnot:** Maps Carnot's verification cascade directly to the TTS taxonomy.
  Carnot implements "energy-based process verification with hardware-accelerated sampling"
  — a distinct point in the taxonomy. Position paper Section 2 should cite this survey to
  frame Carnot's contributions. The open problems it identifies (generalization, efficiency)
  are exactly what Carnot's FPGA path and SOS-KAN certified energy address.
- **Concrete use:** Position paper Section 2 (related work) — cite as taxonomy anchor.
  Show where Carnot sits in the survey's 2D space (energy-based × hardware-accelerated).
- **When to incorporate:** Position paper v2 (Exp 1078).

## 2026-04-30 arxiv Scan (Milestone 2026.04.83 Planning)

### Semantic Energy: Detecting LLM Hallucination Beyond Entropy
- **Paper:** arXiv 2508.14496 (August 2025)
- **What:** Combines semantic clustering with a Boltzmann-inspired energy distribution to detect
  LLM hallucinations. Operates directly on penultimate-layer logits (pre-softmax) rather than
  post-softmax probabilities. Key insight: semantic entropy fails when softmax compression hides
  the model's true uncertainty; Semantic Energy bypasses this by reading the raw logit energy.
  Reports significant improvements over semantic entropy on hallucination detection benchmarks.
- **Relevance to Carnot:** This is an independent derivation of Carnot's core architectural claim:
  energy-based signals (pre-softmax logits) contain more information about model uncertainty than
  probability-based signals (post-softmax). The Boltzmann framing maps directly to Carnot's Ising
  energy tier. Critically, this paper validates that our Tier 0c NUP Probe v4 (bigram dot product
  on logits) is on the right track — both papers extract energy from pre-normalized representations.
  The "beyond entropy" framing is the right vocabulary for the position paper: cite as concurrent
  work with complementary experimental validation.
- **Concrete use:** Position paper Section 2 (related work): cite alongside Eidoku (2512.20664)
  as contemporaneous energy-based verification approaches. Also: the logit-based energy signal
  could enhance Tier 0b (SpilledEnergyDetector, arXiv 2602.18671) by operating on penultimate
  layer logits rather than output logit discrepancy alone. Potential Exp 1080+ SemEnergy probe.
- **Phase relevance:** Phase 1 (validates Tier 0b/0c architecture); position paper.
- **When to incorporate:** Position paper draft (Exp 1075); Tier 0 probe ensemble as stretch goal.

### Decomposing Large-Scale Ising Problems on FPGAs: A Hybrid Hardware Approach
- **Paper:** arXiv 2602.15985 (February 2026)
- **What:** Heterogeneous system: FPGA-based decomposer tightly integrated with a custom 28nm
  Ising solver chip. Key result: nearly 2x speedup and 100x+ energy efficiency vs optimized CPU
  software by co-locating problem decomposition with the solver, eliminating host-device latency.
  Enables solving problems with "thousands of variables" — well beyond single-chip Ising scales.
- **Relevance to Carnot:** Phase 2 hardware architecture. The FPGA-as-decomposer pattern is
  exactly what Carnot's KV260 path implements implicitly: the ARM cores decompose the problem,
  the FPGA PL runs the Ising sampler. This paper validates that the decomposition bottleneck is
  real and the co-design approach achieves the expected gains. The 100x energy efficiency number
  can be cited in the position paper hardware section as near-term-achievable targets.
  More importantly: the "thousands of variables" scope aligns with Carnot's Phase 2 mandate
  (1k-10k p-bits on KV260). We are not trying to solve general MAX-3SAT; we are solving
  structured constraint verification problems where the decomposition can be pre-computed.
- **When to incorporate:** Position paper Section 5 (hardware path); hardware wishlist update.
  No new experiment needed — validates existing Phase 2 design decisions.

### Self-Distilled RLVR: Closing the FR-11 Loop Without External Verifier
- **Paper:** arXiv 2604.03128 (April 2026)
- **What:** Combines RLVR (verifier-guided RL) with on-policy self-distillation. The verifier
  provides a learning signal; self-distillation prevents the model from collapsing to the
  verifier's null space. Key result: RLVR+SD outperforms RLVR alone by +7pp on math benchmarks
  with no additional human labels. The paper explicitly shows that self-distillation without
  a verifier collapses in 3-5 rounds (Zenil Theorem 4 empirically confirmed).
- **Relevance to Carnot:** Direct empirical validation of the Phase-3 architecture chain.
  Zenil's Theorem 4 (α_t → 0 implies collapse) is confirmed experimentally: pure self-distillation
  without RLVR signal collapsed after 3-5 rounds. This paper's combination — RLVR (verifier
  as α_t μ_P term) + self-distillation (student-teacher gap) — is precisely Carnot's FR-11 target.
  The +7pp improvement over verifier-alone also suggests that combining Energy-Selection SSD
  (Carnot's current FR-11 attempt) with a step-level RLVR signal could close the remaining gap.
  Use in Exp 1074 (FR-11 alpha_t v3) as the theoretical motivation for combining energy-selection
  with RLVR signal rather than treating them as alternatives.
- **Phase relevance:** Phase 3 (self-distillation architecture); FR-11 mandatory experiments.
- **When to incorporate:** Exp 1074 (FR-11 alpha_t live v3) — cite as empirical grounding for
  combining Carnot verifier signal with self-distillation. Position paper Section 3.

### Embarrassingly Simple Self-Distillation Improves Code Generation
- **Paper:** arXiv 2604.01193 (April 2026)
- **What:** Shows that a minimal self-distillation recipe (generate N candidates, filter by
  correctness, fine-tune on filtered set) improves code generation by +3-8pp on HumanEval/LiveBench
  without any external judge — just execution feedback as the filter. Key: the correctness filter
  IS the α_t μ_P term; the method works precisely because execution is a ground-truth oracle.
- **Relevance to Carnot:** Carnot's code verification path (SymCodeVerifier + execution feedback,
  Tier 2.5) implements this exact filter, but without the fine-tuning step. This paper suggests
  that adding a distillation loop over execution-verified code generations would close the
  HumanEval gap from +3pp (Carnot's live result) to +6-11pp. The "embarrassingly simple" framing
  is also useful for the position paper: Carnot's energy-based filter is a generalization of
  execution feedback to non-executable domains (math, logic, prose).
- **When to incorporate:** Exp 1074 (FR-11 self-learning) or a follow-on code SSD experiment.
  Position paper Section 4 (architecture) — cite as supporting evidence that filtered SSD works.

### Kaiwu: Bridging Deep Learning and Photonic Quantum Computing for EBMs
- **Paper:** arXiv 2602.19114 (February 2026)
- **What:** PyTorch plugin that bridges deep learning workflows with photonic quantum computing
  hardware for energy-based model inference. Implements the backend protocol needed to run EBM
  inference on photonic processors (similar to how PennyLane bridges to quantum hardware).
  Uses interference-based sampling for EBM energy evaluation rather than MCMC.
- **Relevance to Carnot:** Phase 2/3 hardware path. Carnot's `SamplerBackend` protocol
  (already in place) is designed for exactly this kind of pluggable backend. A Kaiwu-backed
  sampler would provide photonic-speed EBM inference (speed-of-light) — the Phase 3 hardware
  tier in `_bmad/architecture.md`. The PyTorch plugin pattern (not a full fork) is also the
  right integration model: wrap Kaiwu as a `SamplerBackend` subclass, keep the rest of the stack.
  This is less speculative than Extropic's XTR-0 (vaporware until shipped); Kaiwu provides a
  real integration path to photonic hardware TODAY. Check Kaiwu's hardware availability.
- **Phase relevance:** Phase 2 (photonic backend for Ising sampling); Phase 3 foundation model.
- **When to incorporate:** Hardware wishlist (add to Priority 3 section). No experiment until
  photonic hardware is accessible.

## 2026-04-29 arxiv Scan (Milestone 2026.04.82 Planning)

### Eidoku: A Neuro-Symbolic Verification Gate for LLM Reasoning
- **Paper:** arXiv 2512.20664 (December 2025)
- **What:** Proposes a constraint-based deterministic verification gate that rejects LLM
  reasoning outputs containing logical or arithmetic contradictions, without any learned
  classifier. Uses deterministic rule-evaluation over a parsed constraint graph derived from
  the chain-of-thought steps. Reports near-zero false-positive rate on benchmarks where
  the constraint encoding is correct.
- **Relevance to Carnot:** Direct architectural parallel to Carnot's Tier 0 verifier cascade.
  The key design decision — parse CoT into a constraint graph, evaluate deterministically —
  is exactly what VeriCoTStepValidator does (Exp 453), but Eidoku scales this to larger
  reasoning tasks. The paper's "neuro-symbolic gate" framing is a useful vocabulary for the
  position paper: Carnot's cascade IS a neuro-symbolic verification gate at inference time.
  Cite in Section 2 (related work) and Section 4 (architecture) of the position paper.
- **Phase relevance:** Phase 1 (current verify-repair) — validates architecture choice; Phase 3
  position paper — cite as contemporaneous approach with architectural comparison.
- **When to incorporate:** Position paper draft (Exp 1063+); no new experiment needed.

### Neural Sum-of-Squares: Certifying Nonnegativity with Transformers
- **Paper:** arXiv 2510.13444 (October 2025)
- **What:** Combines transformer-based function approximation with Sum-of-Squares (SOS) certificates
  for certifying polynomial nonnegativity. Uses learned Gram matrices to construct valid SOS
  decompositions. Demonstrates that transformers can efficiently parameterize the Gram matrix
  while maintaining the SOS structure (Gram is PSD ↔ polynomial is SOS). Empirically faster
  than pure SDP solvers on high-degree polynomials.
- **Relevance to Carnot:** Direct complement to SOS-KAN (Exp 1047). Where SOS-KAN uses a fixed
  SOS parameterization for the derivative ψ'(x), this paper suggests using a learned Gram matrix
  for the SOS decomposition — potentially more expressive while still certifiably nonnegative.
  The transformer Gram matrix idea could replace V·V^T in SOSKANEnergy with a learned
  low-rank PSD approximation. More importantly: the paper's certification framework could be
  used to produce FPGA-ready SOS certificates for the Ising coupling matrix J, closing the
  loop between energy certification and hardware deployment.
- **Concrete experiment:** SOS-KAN v3 (milestone .83+) — replace V·V^T with a learned Gram
  matrix from a small transformer. Compare expressivity vs. SOSKANEnergy v1 (Exp 1047)
  on the expanded FoVer corpus. Target: AUROC >= 0.72 with certifiable invariants.
- **When to incorporate:** Milestone .83 (after expanded FoVer corpus is available from .82).

### Neural Uncertainty Principle: Unified View of Adversarial Fragility and LLM Hallucination
- **Paper:** arXiv 2603.19562 (March 2026)
- **What:** Frames LLM hallucination and adversarial fragility as two manifestations of the same
  underlying phenomenon: learned representations that violate geometric constraints in the model's
  implicit constraint space. Proves a "neural uncertainty principle" — the tradeoff between a
  model's ability to satisfy multiple constraint types simultaneously is bounded by an information-
  theoretic quantity related to the model's representational capacity. Key result: hallucinations
  concentrate in the region where constraint satisfaction probability drops below a threshold θ*,
  providing a testable prediction.
- **Relevance to Carnot:** Provides theoretical grounding for why energy-based verification works.
  Carnot's energy function IS the constraint-violation detector the paper describes. The θ*
  threshold maps to Carnot's energy threshold for "skip tier" decisions. The concentration
  result suggests that hallucinations are not random — they cluster near specific constraint
  boundaries, which Carnot's Ising energy landscape can learn to model.
  The adversarial angle is directly relevant to RETRO-031 (adversarial degradation): if the
  adversarial perturbation is designed to push examples across the θ* boundary, that explains
  why Carnot's verify-repair showed adversarial DROP but not recovery — the repair step needs
  to push the representation BACK across θ*.
- **Phase relevance:** Phase 1 (explains adversarial behavior); Phase 3 position paper
  (theoretical framing section — cite as independent derivation of why EBM verification works).
- **When to incorporate:** Position paper Section 3 (theoretical foundations). Potential future
  experiment: measure θ* empirically using Carnot's energy scores on the FoVer corpus — does
  hallucination concentration hold?

### 250 Magnetic Tunnel Junctions-Based Probabilistic Ising Machine
- **Paper:** arXiv 2506.14590 (June 2025)
- **What:** Hardware Ising machine using 250 Stochastic Magnetic Tunnel Junctions (STT-MTJs)
  on a custom CMOS chip. Achieves ~10x better energy efficiency than GPU-based Ising solvers
  at equivalent problem sizes. Key technique: thermal noise from STT-MTJ junctions directly
  implements the p-bit stochastic flip without a random-number generator circuit.
  Demonstrated on combinatorial optimization up to 250 spins.
- **Relevance to Carnot:** Phase 2 hardware path. The SamplerBackend abstraction (Exp 71)
  already supports pluggable hardware backends. An MTJ-based backend would provide
  thermodynamic sampling at hardware-native noise temperatures — closer to the Extropic Z1
  dream than the KV260 digital Ising. Key numbers: 250 spins × 10x energy efficiency vs GPU
  puts this in the same neighborhood as D-Wave for small constraint problems. The CMOS
  implementation is more accessible than D-Wave QPU time (~$2000/hr). Worth tracking for
  Phase 2 hardware procurement.
- **Strategic note:** Validates Extropic's hypothesis that thermodynamic noise IS computation
  for Ising machines. Carnot should cite this in the hardware track of the position paper
  as evidence that the thermodynamic computing path is not speculative.
- **When to incorporate:** Hardware wishlist (add to Priority 5 section); cite in position paper
  hardware section. No new experiment needed until Phase 2 hardware arrives.

### KAN Applied to Crystal Energy Landscape Interpretation
- **Paper:** arXiv 2604.04636 (April 2026)
- **What:** Applies Kolmogorov-Arnold Networks to interpret crystal energy landscapes in
  computational materials science. KAN splines learn per-feature energy contributions
  with explicit interpretability — each spline can be visualized as a 1D potential curve.
  Key finding: KAN extrapolates more reliably than MLP outside the training distribution
  for energy-dense configurations, because its activation functions are learned from data
  rather than fixed.
- **Relevance to Carnot:** Independent validation that KAN is a strong architecture for
  energy function approximation. The extrapolation finding is directly relevant to
  Carnot's concern about FoVer corpus coverage — KAEMEnergy and GS-KAN may generalize
  better to unseen constraint types than MLP-based alternatives. The paper's interpretability
  angle (visualizable per-spline contributions) could make Carnot's verifier more
  explainable: "this constraint fired because feature X exceeded threshold Y" rather
  than a black-box energy score.
- **When to incorporate:** Background reference for position paper Section 4 (architecture).
  Consider adding a "verifier interpretability" section to the position paper based on this.

## Study Sources & Discovery Tools

Meta-sources used to identify papers, repos, and tools worth evaluating.
Not content itself, but signals to prioritise what to read next.

### Gerolamo — Technical Intelligence Corpus
- **URL:** https://gerolamo.org/
- **What:** Searchable, scored index of open-source repos, arXiv papers, and HuggingFace
  models. Proprietary metrics include defensibility, threat profile, composability, and
  market-concentration/"deep moat" ratings, layered on top of star/fork counts. Covers AI
  frameworks + transformer models, cryptography (incl. post-quantum), blockchain smart-
  contract languages, and open-source infrastructure. Created by Adjective (adjective.us).
  Free tier browses top-rated entries; premium tier adds workspaces, AI-assisted
  composition, and custom corpus building.
- **How Carnot should use it:** Complementary to arXiv scans — arXiv surfaces individual
  papers; Gerolamo scores *adoption + defensibility signals at the repo level*. Useful for:
  (1) vetting repos before citing them (see OpenMythos case: 9.3k stars but unvetted — a
  moat/defensibility score would have flagged it); (2) discovering emerging hardware and
  ML-framework projects relevant to Phase 2/3 (FPGA/Ising, EBT, photonic); (3) tracking the
  competitive landscape around EBM tooling (thrml, EB-JEPA, TorchEBM).
- **Caveats:** Scoring methodology is proprietary — treat "deep moat" rating as one signal,
  not ground truth. Author "Adjective" is not a known institutional research source; the
  value is in the aggregation + filter, not any editorial voice. Premium-gated features may
  limit autonomous-conductor use unless an account is provisioned.
- **When to consult:** Milestone planning — alongside the arXiv scan, spot-check Gerolamo's
  top-rated entries in EBM / post-quantum / AI-framework categories for repos we haven't
  seen. Also useful mid-experiment when deciding whether to invest in integrating a
  third-party tool (use the defensibility/composability score as a risk input).

## 2026-04-29 user-flagged Apple ParaRNN Research

### ParaRNN: Large-Scale Nonlinear RNNs Trainable in Parallel (Apple 2025/2026)
- **URL:** https://machinelearning.apple.com/research/large-scale-rnns
- **Code:** https://github.com/apple/ml-pararnn (Apache-licensed)
- **What:** Newton's method applied to nonlinear RNN recurrence equations,
  reducing each step to a parallelizable linear system (~3 Newton iterations).
  ParaGRU / ParaLSTM use diagonal / block-diagonal gate matrices yielding
  structured Jacobians for cheap inner solves. Three-tier implementation
  (PyTorch autodiff → CUDA kernels for structured Jacobians → fully-fused
  single-kernel Newton routine). 400M-7B parameter models reach transformer
  parity in perplexity AND surpass linear SSMs (Mamba) on state-tracking
  and recall tasks.
- **Strategic implication for Carnot:** the 13-contribution math chain
  (Zenil + Kinematic + Ising-Rank) proved verifier-rotation architecture
  is provably optimal for Boolean verifiers + smooth EBM core. *It does
  not constrain what the smooth EBM core is.* We've been implicitly
  assuming energy-based *transformers*. ParaRNN suggests energy-based
  *recurrent models* may be the more natural Phase 3 substrate:
  constant-cost-per-token generation, better recall than Mamba,
  diagonal Jacobians map cleanly to Phase 2 transpiler's local Ising
  couplings. **Phase 3 architectural decision deferred to a future
  proposal: transformer-EBT vs recurrent-EBT.**
- **Newton-iteration trick is reusable for EBT energy descent.** Newton
  solves $\nabla E = 0$ — mathematically identical template to RNN
  hidden-state recurrence. Carnot can adopt the parallelization template
  directly for EBT inference.
- **Strengthens Hybrid Coprocessor Pipeline (Round-13).** Structured-
  Jacobian discipline + diagonal hidden state is exactly what the
  FPGA-side smooth EBM core wants.
- **Lineage gap surfaced:** prior work on Newton/quasi-Newton parallel
  RNN training that Carnot didn't track:
  - **DEER (Lim et al. 2024)**: Differential Equation-style RNN
    parallelization via Newton iteration.
  - **ELK (Gonzalez et al. 2024)**: extension of DEER with broader
    nonlinear classes.
  Carnot's literature-priority discipline missed these. Should be
  audited per the `feedback_literature_priority_discipline` memory.
- **Touch-points:** long-context reasoning (constant-cost generation),
  continuous latent space (relevant to EBT vision), hardware
  acceleration (CUDA tiers + structured Jacobians map to FPGA).
  NOT relevant to: self-distillation / α_t grounding (orthogonal),
  Phase 1 verify-repair (orthogonal).
- **Decentralization status:** Apache-licensed open source; satisfies
  rules 1 (local-first) and 3 (mirroring) once we add it to the
  Carnot-tracked dependency set.

## 2026-04-28 user-flagged Self-Improvement Limits Theory

### On the Limits of Self-Improving in LLMs (Zenil 2026)
- **Paper:** arXiv 2601.05280v2 — "On the Limits of Self-Improving in Large Language Models: The Singularity Is Not Near Without Symbolic Model Synthesis"
- **Author:** Hector Zenil
- **What:** Formalises recursive self-training as a dynamical system μ_{t+1} = (1−α_t)μ_t + α_t μ_P + ξ_t where μ_P is the true distribution and α_t is the proportion of exogenously-grounded data per round. Theorem 2: 𝔼[H(Q_{t+1})] ≤ H(Q_t) − Δ(N) (entropy strictly decreases). Theorem 4: as α_t → 0 the model becomes a random walk. Theorem 5: convergence to truth P requires inf_t α_t > 0 (a non-vanishing grounding signal). Escape route: Coding Theorem Method m̂_{CTM}(o) = (1/|M|) Σ 𝟙{U_M↓=o} and Block Decomposition Method BDM_k(o) = Σ_i CTM(b_i) + log n_i, which break the Data Processing Inequality via the universal prior m(p) ∝ 2^{−|p|}.
- **Why this matters for Carnot:** This is the formal mathematical statement of why Carnot's verifier exists. Theorem 4's `α_t μ_P` term IS what an external verifier provides to a self-distilling LLM. The paper gives Phase 3's foundation-model architecture a falsifiable prediction: any training loop where the verifier signal vanishes will collapse — measurably.
- **Phase relevance:**
  - Phase 1 (verify-repair): legitimises the verifier-as-grounding role; Carnot's energy function = α_t μ_P contribution.
  - Phase 3 (foundation model): foundational. Any Carnot self-improvement loop (autoresearch, EBT pretraining) must track α_t as a first-class metric.
- **Adversarial check vs SSD:** Apple SSD self-distillation (already in MEMORY) claims +12.9pp on LiveCodeBench *without* an external verifier. Zenil predicts this either silently relies on residual α_t > 0 from the base model's pretraining, or will collapse over multiple rounds. Worth instrumenting: replicate SSD over N rounds and measure KL(Q_t ‖ P) drift.
- **Reusable primitives:**
  - **BDM_k as a complexity-energy term:** add E_complexity(x) = BDM_k(x) to Carnot's repair scorer for tie-breaking when energies are degenerate. Tractable proxy for Kolmogorov complexity over discrete outputs (code, proofs).
  - **α_t accounting in autoresearch:** instrument the conductor's self-improvement loop so each milestone reports the α_t (verifier-grounding ratio) for any self-distillation it ran. Doomed-rerun discipline already retires repeating failures; α_t < threshold should be a separate retire trigger for self-distillation experiments.
  - **Theorem 5 contractivity bound:** falsifiable test for any future Carnot training loop — measure KL(Q_t ‖ P) against α_t and verify the predicted regime. If empirical drift contradicts Zenil's bound, that's a publishable counterexample.
- **Two-source check:**
  - **Builds on:** Shumailov et al. 2023/2024 (model collapse, empirical), Alemohammad et al. 2023 (MAD self-consuming generative models), Gerstgrasser et al. 2024 (real/synthetic ratio).
  - **In tension with:** Apple SSD (no verifier, claimed gains), various "self-rewarding" approaches. Worth fast-following Shumailov 2024 and Alemohammad 2023 next; both are empirical baselines to which Zenil's theorem can be applied.
- **How Carnot should use it:**
  - Cite in any future paper that motivates the verifier role.
  - Add α_t metric collection to the autoresearch loop (tie into ops/metrics.md or a new α_t.csv).
  - Audit any planned SSD-style experiment against Theorem 5 — if the experiment cannot articulate where α_t comes from, the experiment is doomed.
- **Caveats:** The constructive escape route (CTM/BDM) requires symbolic model synthesis machinery Carnot does not currently have; the paper's prescriptive arc would push Carnot toward a hybrid energy-+-CTM-scorer, which is a substantive Phase 3 architectural choice, not a trivial integration.

## 2026-04-28 arxiv Scan (Milestone 2026.04.79 Planning)

### Multilevel Training for Kolmogorov-Arnold Networks
- **Paper:** arXiv 2603.04827 (March 2026)
- **What:** Achieves orders-of-magnitude accuracy improvements through multilevel training with
  spline knot refinement — coarser grids first, then progressively refined. Particularly
  effective for physics-informed problems and small datasets. Compatible with existing KAN 2.0
  spline activations.
- **Relevance to Carnot:** Carnot's KAEMEnergy and GS-KAN energy tier are trained on small
  FoVer corpora (57–500 pairs). Multilevel training should reduce epochs-to-convergence by
  reducing early-training instability. Combines naturally with the Newton-Kaczmarz optimizer
  (arXiv 2512.18921): NK handles second-order convergence while multilevel handles grid structure.
  Together they could reduce KAN training overhead from ~15 min to ~2 min per experiment, directly
  improving autoresearch iteration speed.
- **Concrete experiment:** Milestone 2026.04.79+ — Integrate multilevel training into
  KAEMEnergy.fit() with three grid levels (G=4→8→16). Compare epochs-to-AUC-0.95 vs
  single-level G=10 training. Target: 3x or more convergence speedup on 57-pair FoVer corpus.
- **When to incorporate:** Milestone 2026.04.79 (Exp 1021, NK-KAEMEnergy + Multilevel).

### Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks
- **Paper:** arXiv 2604.03345 (April 2026)
- **What:** Derives platform-independent complexity metrics for KAN hardware: Rational
  Multiplications (RM), Bit Operations Per weight (BOP), and Normalized Activation Bit
  Sparsity (NABS). Validated on FPGA implementations. Provides closed-form formulas for
  estimating KAN LUT, DSP, and BRAM usage from architecture hyperparameters without needing
  synthesis.
- **Relevance to Carnot:** Carnot's KV260 FPGA target has a hard 117K LUT budget. The current
  approach to LUT estimation is heuristic (param_count × multiplier). This paper provides
  architecture-level formulas: LUT ≈ RM × bits_per_weight, where RM = n_inputs × n_hidden ×
  G (grid points). For KAEMEnergy(n=64, h=8, G=10): RM = 64×8×10 = 5120 rational mults. At
  16-bit precision, estimated LUT ≈ 5120×16 = 82K — just under the 117K budget. For GS-KAN
  with shared parent (G=1 per edge, N_transforms=1): RM = 64×8×1 = 512, LUT ≈ 8K — very
  comfortable fit. This paper lets us derive GS-KAN's architectural advantage analytically.
- **Concrete experiment:** Milestone 2026.04.79 — Add RM/BOP/NABS computation to GS-KAN v3
  experiment (Exp 1019). Compare KAEMEnergy vs GS-KAN on hardware complexity metrics, not just
  parameter count. Target: GS-KAN RM < 20% of KAEMEnergy RM while maintaining AUROC ≥ baseline.
- **When to incorporate:** Milestone 2026.04.79 (Exp 1019, GS-KAN v3).

### Digitally Optimized Initializations for Fast Thermodynamic Computing
- **Paper:** arXiv 2603.24183 (March 2026)
- **What:** Uses digital processors to compute Mpemba-optimized initializations for thermodynamic
  computing hardware. The "Mpemba effect" for spin systems: starting from a specific correlated
  initial state reaches the target equilibrium faster than random initialization, suppressing
  slow relaxation modes. Demonstrates 2-5x convergence speedup on hardware Ising machines without
  changing the hardware itself.
- **Relevance to Carnot:** Carnot's ALMC-ODE annealed sampler (arXiv 2604.20052) and KV260 FPGA
  Ising sampler both use random initialization, which causes slow convergence on large spin counts
  (the primary reason SpilledEnergy AUROC was 0.5 on live data — insufficient annealing steps).
  Mpemba initialization is purely digital (compute on CPU, upload to FPGA/sampler) and requires
  no hardware changes. For the KV260, this means: compute the Mpemba-optimal initial configuration
  on CPU (from the mean-field equations), AXI-upload it to the Ising core instead of random init.
  Expected convergence: 2-5x faster per constraint check, which translates to 2-5x lower latency
  or 2-5x more samples in the same budget.
- **Concrete experiment:** Milestone 2026.04.79 — Add Mpemba initialization to ALMCODESampler
  (Exp 1020). Compute mean-field equilibrium as warm start, compare convergence steps vs random
  init. Target: convergence steps < 50% of random-init baseline on bimodal problem.
- **When to incorporate:** Milestone 2026.04.79 (Exp 1020, ALMC-ODE v2).

### Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs
- **Paper:** arXiv 2601.18734 (January 2026)
- **What:** Single LLM acts as both teacher (conditioned on verified reasoning traces) and student
  (standard context). Achieves self-improvement with only 1% of the process labels required by
  standard PRMs. Superior token efficiency compared to RLVR alone. Key mechanism: the "teacher"
  mode sees verification outcomes (correct/incorrect step labels) that the "student" mode does not.
- **Relevance to Carnot:** Energy-Selection SSD (FR-11 mandatory self-learning) is Carnot's
  version: the "teacher" is the energy function selecting high-confidence outputs. This paper
  provides a complementary mechanism: use Carnot's step-level FoVer labels as the "teacher"
  conditioning signal rather than just energy thresholds. Combining Self-Distilled Reasoner's
  label-conditioned distillation with Energy-Selection SSD's energy filter creates a richer
  self-improvement signal: (1) energy threshold removes low-confidence outputs, (2) step labels
  condition the teacher on WHICH steps were verified correct. This may close FR-11 more
  thoroughly than either approach alone.
- **Concrete experiment:** Milestone 2026.04.79 — Include Self-Distilled Reasoner as a comparison
  baseline in Energy-Selection SSD v2 (Exp 1015). Measure: energy-only filter vs label-conditioned
  filter vs combined. If combined approach outperforms energy-only, document as FR-11 closure path.
- **When to incorporate:** Milestone 2026.04.79 (Exp 1015, Energy-Selection SSD v2).

### Necessary and Sufficient Conditions for Universality of KAN
- **Paper:** arXiv 2604.23765 (April 2026)
- **What:** Proves that a single non-affine function suffices for universal approximation in deep
  KANs (more than one hidden layer). Provides tight lower bounds on the number of hidden units
  required for universality. Resolves a theoretical gap in prior KAN universality results.
- **Relevance to Carnot:** Carnot's KAEMEnergy uses KAN as a universal verifier for constraint
  satisfaction. This paper provides formal guarantees: a 2-layer KAN with a single non-affine
  activation CAN represent any constraint energy function. This is the theoretical underpinning
  for why KAN is a valid energy tier architecture (not just empirically observed — provably
  sufficient). The lower bound on hidden units provides a principled way to set n_hidden in
  KAEMEnergy: for FoVer's 57-pair corpus with ~10 effective constraint types, the bound
  suggests n_hidden ≥ 8 is sufficient for universality.
- **When to incorporate:** Background reference. Cite in spec update when adding REQ-SAMPLE-020+
  (GS-KAN and NK variants). No new experiment needed — validates existing architecture.

### LagONN: Lagrange Oscillatory Neural Networks for Constraint Satisfaction
- **Paper:** arXiv 2505.07179 (May 2025)
- **What:** Introduces additional oscillators as Lagrange multipliers to guide Hopfield-style
  networks toward feasible regions in the energy landscape. Avoids infeasible attractors by
  augmenting the energy function with constraint-penalty terms that adapt at runtime. Key
  advantage over standard penalty methods: oscillators provide continuous feedback, not
  step-function penalties — gentler landscape modification.
- **Relevance to Carnot:** PPSEBM (Progressive Penalty Self-Evolving Boltzmann Machine) already
  uses an adversarial penalty mechanism. LagONN's oscillatory Lagrange approach could improve
  PPSEBM's ability to escape infeasible constraint configurations. More importantly: for the
  FoVer constraint corpus, LagONN's "infeasibility detection via oscillator phase" is structurally
  identical to Carnot's RETRO-CONSTRAINT-ZERO-DELTA root cause — when constraints are orthogonal
  (semantically non-overlapping), the penalty landscape is flat. LagONN's Lagrange oscillators
  explicitly handle this case by adding feedback when delta=0.
- **Concrete experiment:** Consider for Milestone 2026.04.80+ — LagONN-inspired PPSEBM: add
  Lagrange oscillators to PPSEBM's constraint penalty mechanism to handle the zero-delta case.
  Target: PPSEBM sessions_with_new_templates > 3 on heterogeneous constraint corpus.
- **When to incorporate:** Milestone 2026.04.80 (after PPSEBM v4 validates baseline).

## 2026-04-27 user-flagged Apple Research

### SSD: Embarrassingly Simple Self-Distillation Improves Code Generation
- **Paper:** arXiv 2604.01193 (April 2026)
- **Authors:** Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert,
  Yizhe Zhang (Apple Research).
- **What:** Sample model's own outputs under specific temperature + truncation settings, then
  apply standard SFT on those samples. No external verifier, no teacher model, no RL. Headline
  result: Qwen3-30B-Instruct improves from 42.4% to 55.3% pass@1 on LiveCodeBench v6 — a 12.9-pp
  jump from self-distillation alone. Mechanism framed as resolving a "precision-exploration
  conflict in LLM decoding": suppress distractor tails where precision matters, preserve diversity
  where exploration matters. Generalizes across Qwen and Llama at 4B/8B/30B, instruct and thinking
  variants, with gains concentrating on harder problems.
- **Relevance to Carnot:** Adversarial baseline AND complementary technique.
  - **Adversarial:** Anywhere we report "Carnot improves the base model by X pp on math/code,"
    we should also measure "SSD-improved base model alone vs. SSD + Carnot." A 12.9-pp jump from
    SSD alone means our claimed deltas may be partially recoverable without verification. Failing
    to separate the two lets us over-claim Carnot's contribution.
  - **Complementary:** SSD reshapes the *generator* distribution; Carnot adds a *verifier* on top.
    The two stack — SSD-improved generator + energy verification + repair could plausibly hit
    higher numbers than either alone.
  - **Cascade structural map:** SSD's "suppress distractor tails / preserve exploration diversity"
    is what our Tier 0/1/2/3/4 cascade does at inference time via energy verification. SSD does
    it at training time via temperature-truncation distillation.
  - **FR-11 + Energy-Selection SSD opportunity:** FR-11 already crawls cross-session memory of
    cascade verdicts. A Carnot-native SSD variant would distill the generator on outputs that
    Carnot's *energy function* marks as high-confidence-correct — using the energy function as
    the SSD selection filter rather than just temperature/truncation. More principled signal
    than the SSD paper has access to.
  - **Math-repair ceiling re-think:** Exps 942/953/963 hit ceilings on SVAMP/GSM8K. Some of that
    may be a generator-distribution problem, not a repair-mechanism problem. Distilling our SOTA
    generators (Qwen3.6-35B-A3B, Gemma-4-31B/26B-A4B) before running repair could partially
    close the gap.
- **Caveats / unknowns:**
  - "Specific temperature and truncation settings" — abstract doesn't disclose them. Reproducing
    requires the full paper.
  - Single benchmark (LiveCodeBench), single-domain (code). Whether the result generalizes to
    SVAMP/GSM8K/FoVer math reasoning is empirically open.
  - SSD-improved models may need Carnot *more*, not less: if SSD over-fits to "model-confident"
    patterns, hallucination on out-of-distribution inputs could *increase* — exactly where our
    verification has highest value.
  - No error bars reported in the abstract; multi-seed validation needed before trusting the
    42.4 → 55.3 number as settled.
- **Three concrete operational next steps** (filed for .78 or later):
  1. Add SSD baseline to the eval pipeline. New experiment: base model vs SSD-distilled vs
     SSD + Carnot, on a held-out math benchmark. Without this comparison our pp-improvement
     numbers are ambiguous.
  2. **FR-11 + Energy-Selection SSD experiment.** Use Carnot's energy function as the SSD filter,
     not temperature/truncation. Train on cascade-high-confidence outputs, evaluate on held-out
     hallucination corpus.
  3. **Math-repair re-think for Phase 1.** Before declaring SVAMP/GSM8K a "ceiling," distill the
     SOTA-tier generators (Qwen3.6-35B, Gemma-4-31B) on their own correct samples and re-measure.

## 2026-04-27 arxiv Scan (Milestone 2026.04.76 Conductor Scan)

### PCIB: Predictive Coding and Information Bottleneck for Hallucination Detection
- **Paper:** arXiv 2601.15652 (January 2026)
- **What:** Combines neuroscience-inspired Predictive Coding signals (surprise against internal priors)
  with Information Bottleneck theory to detect LLM hallucinations. Achieves 0.8669 AUROC using a
  <1M parameter classifier trained on 75x less data and running 1000x faster than LLM-judge baselines.
  Introduces a "Falsifiability Score" that detects when the model states confident claims that contradict
  the source context — a signal unavailable from token-probability alone.
- **Relevance to Carnot:** The predictive coding surprise signal is structurally identical to Carnot's
  energy: a low-surprise (low-energy) token fits the constraint manifold; a high-surprise token is an
  anomaly. The Falsifiability Score maps directly onto Carnot's Phase 1 goal — detecting outputs that
  are confident AND wrong. Key advantage over entropy-based detectors (Semantic Entropy, arXiv 2508.14496):
  PCIB is interpretable (signals have causal explanations), operates without a retrieval corpus, and is
  fast enough for inline deployment at inference time. The paper's negative finding — "Rationalization
  signal fails to distinguish hallucinations" — confirms that asking the model to explain itself is not
  a reliable verification strategy, validating Carnot's energy-grounded (rather than self-report-grounded)
  approach.
- **Concrete experiment:** Milestone 2026.04.77+ — PCIB signals as EBM input features: extract
  entity-focused uptake and falsifiability scores from a local Gemma4-E4B-it model, train a KAN energy
  function on these features, compare AUROC vs raw embedding baseline (Exp 442/944 corpus).
- **When to incorporate:** Milestone 2026.04.77 (after ThreeTierPipeline baseline from .75/.76).

### Concurrent KAN Training via Disjoint Datasets with FPGA Parallelization
- **Paper:** arXiv 2512.18921 (December 2025)
- **What:** Presents three acceleration techniques for KAN training under the Newton-Kaczmarz (NK)
  optimizer: (1) a pre-training initialization aligned with NK's update structure, (2) disjoint dataset
  training with model merging (trains on separate data subsets independently then combines), and (3) FPGA
  hardware parallelization of basis function evaluation. Reports KANs training "more than 40 times faster
  than neural networks" at equivalent accuracy on CPU; FPGA implementation is fully reproducible with
  open-source code.
- **Relevance to Carnot:** Carnot's KAEMEnergy (KAN-based energy tier) is trained offline and then
  deployed on KV260 FPGA for inference. This paper provides two complementary benefits: (a) the 40x CPU
  speedup directly reduces experiment iteration time for training new constraint-type KANs in the
  autoresearch loop — each experiment currently spends ~15min on KAN training overhead; (b) the FPGA
  parallelization technique directly informs the KV260 v4 RTL design for KAN inference (hardware/kv260/)
  — the disjoint-dataset merging is the software analogue of the hardware's pipelined basis evaluation.
  The NK convergence is also more suitable than Adam for Carnot's small-N constraint datasets (57-500
  pairs), where second-order methods are not dominated by stochastic gradient noise.
- **Concrete experiment:** Milestone 2026.04.77+ — NK-trained KAEMEnergy: replace Adam optimizer with
  Newton-Kaczmarz in KAN energy training (Exp 936/948 pipeline). Compare training time and AUROC on the
  57-pair FoVer corpus and 500-pair Z3-expanded corpus (from arXiv 2505.15960 expansion plan).
- **When to incorporate:** Milestone 2026.04.77 (requires Z3-expanded corpus from .76 Exp 983 first).

### Scaling Laws for Gaussian Kolmogorov-Arnold Networks
- **Paper:** arXiv 2604.21174 (April 2026)
- **What:** Derives a principled operating interval ε ∈ [1/(G-1), 2/(G-1)] for the Gaussian scale
  parameter in FastKAN/Gaussian KANs, validated across function approximation and physics-informed
  problems. Shows that the first-layer scale determines distinguishability of inputs and cannot be
  recovered by later layers — positioning scale selection as a design-time rather than training-time
  decision. Demonstrates Gaussian KANs are competitive with Chebyshev KANs when properly scaled.
- **Relevance to Carnot:** Carnot's KAEMEnergy uses B-spline activations (standard KAN 2.0). Gaussian
  KANs are already implemented in the carnot-kan crate as an alternative backend (FastKAN path). This
  paper fills the missing design rule: previously, the Gaussian scale parameter was set heuristically
  or grid-searched. With the ε ∈ [1/(G-1), 2/(G-1)] rule, Carnot can initialize the Gaussian KAN
  energy tier analytically based on the number of grid points G, eliminating scale-search overhead
  from the autoresearch loop. The physics-informed problem results (PDEs) are directly analogous to
  Carnot's constraint satisfaction setting where the energy landscape must conform to known invariants.
- **Concrete experiment:** Milestone 2026.04.77+ — Apply the ε design rule to Gaussian KAEMEnergy
  initialization. Compare convergence speed (epochs to AUC=0.95) vs B-spline KAN on 57-pair FoVer
  corpus. Hypothesis: Gaussian KAN reaches target AUC 2-3x faster with analytic scale initialization
  vs grid search or B-spline default.
- **When to incorporate:** Milestone 2026.04.77 (combinable with NK optimizer experiment above).

### Quantum Annealing Algorithms for Estimating Ising Partition Functions
- **Paper:** arXiv 2504.21666 (April 2025)
- **What:** Introduces a quantum protocol combining reverse quantum annealing with optimized
  nonequilibrium initial distributions to estimate Ising partition functions. Dramatically reduces
  estimator variance compared to classical Jarzynski equality approaches, improving performance
  scaling "from ~8.5 to ~0.5" on spin glass benchmarks. Near-term feasibility demonstrated on
  superconducting qubits and trapped ion platforms without requiring strict adiabatic conditions.
- **Relevance to Carnot:** Partition function estimation is the foundation of thermodynamic computing's
  energy calibration: the inverse temperature β in Carnot's Ising sampler is calibrated by estimating
  the partition function Z(β). Currently, Carnot uses heuristic temperature schedules (simulated
  annealing). This paper provides a quantum path to exact Z estimation, relevant to Phase 2 when
  Carnot integrates D-Wave or Pasqal neutral-atom hardware (already in hardware wishlist). More
  immediately: the reverse annealing + optimized initialization pattern is implementable on classical
  hardware as an improved initialization strategy for the KV260 FPGA Ising sampler (currently uses
  random initialization, which is the main cause of slow convergence on large spin counts).
- **Concrete experiment:** Consider for Milestone 2026.04.78+ after KV260 v4 RTL is validated.
  Classical analogue: implement reverse annealing initialization (start from a warm state near the
  expected optimum rather than random) in Python Ising sampler, measure convergence time reduction.
- **When to incorporate:** Milestone 2026.04.78 (Phase 2 hardware track, after KV260 v4 RTL).

### GS-KAN: Parameter-Efficient KANs via Sprecher-Type Shared Basis Functions
- **Paper:** arXiv 2512.09084 (December 2025)
- **What:** Proposes GS-KAN, a parameter-efficient variant of Kolmogorov-Arnold Networks that
  maintains a single shared parent function per layer and constructs edge-specific functions via
  learnable linear transformations. Directly inspired by David Sprecher's constructive proof of
  the Kolmogorov superposition theorem. Enables KAN deployment in high-dimensional settings where
  standard per-edge parameterization causes parameter explosion. Outperforms MLPs on high-dimensional
  classification while maintaining competitive function approximation accuracy.
- **Relevance to Carnot:** Carnot's KV260 FPGA target (117K LUTs, 216 BRAM tiles) has strict
  parameter memory budgets. Standard KAEMEnergy with G=10 grid points × N inputs × hidden units
  already pushes the BRAM budget (see KV260 LUT overflow RETRO-072). GS-KAN's shared parent function
  approach reduces KAN memory by roughly N_edges / 1 per layer — for a 2-layer KAN with 64 inputs
  and 8 hidden units, this is an ~8x reduction in basis storage. Combined with the Newton-Kaczmarz
  training (arXiv 2512.18921), GS-KAN provides both the parameter reduction needed for KV260 fit
  AND the training speed needed for autoresearch iteration. The Sprecher construction also provides
  a formal justification for why KANs generalize well on small Carnot datasets: the shared parent
  function is a universal approximator that the linear transformations specialize — not an overfit
  per-example memorization.
- **Concrete experiment:** Milestone 2026.04.77+ — GS-KAN energy tier: implement GS-KAN in
  python/carnot/models/kan.py as an alternative to standard KAN 2.0 energy tier. Train on FoVer
  57-pair corpus. Compare: (a) AUROC vs standard KAN, (b) parameter count, (c) estimated LUT usage
  via synthesis report. Target: AUROC parity with <50% of standard KAN's parameter count.
- **When to incorporate:** Milestone 2026.04.77 (KV260 memory budget is a hard constraint from .75).

## 2026-04-27 arxiv Scan (Milestone 2026.04.76 Planning)

### Unlocking the Power of Boltzmann Machines by Parallelizable Sampler and Efficient Temperature Estimation
- **Paper:** arXiv 2512.02323 (December 2025)
- **What:** Introduces Langevin Stochastic Boltzmann (LSB) sampler inspired by quantum-inspired
  combinatorial optimization. Provides parallelized Boltzmann sampling with MCMC-comparable accuracy
  using a Langevin-guided update rule rather than sequential Gibbs sampling. Also proposes conditional
  expectation matching (CEM) for efficient temperature estimation. Reported speedups of 3-8x over
  sequential Gibbs at equivalent accuracy on Ising constraint problems.
- **Relevance to Carnot:** Carnot's parallel Ising sampler (183x faster than thrml, Exp 285) uses
  a custom parallel Gibbs scheme. LSB provides a principled alternative: Langevin dynamics naturally
  parallelize across all spins simultaneously, no checkerboard ordering needed. CEM temperature
  estimation replaces manual beta tuning — critical for Tier 1 constraint learning where beta should
  adapt per-domain. FPGA implementation path: Langevin update is a multiply-accumulate operation,
  even more hardware-friendly than conditional Gibbs.
- **Concrete experiment:** Exp 983 — Langevin SB Parallelizable Boltzmann Sampler: replace current
  parallel Ising sampler with LSB on CPU; compare convergence rate and AUROC on the existing
  constraint benchmark suite. Target: any improvement in convergence speed without AUROC regression.
- **When to incorporate:** Milestone 2026.04.76 — Phase 3 (Exp 983).

### Generalizable Process Reward Models via Formally Annotated Step-Level Data
- **Paper:** arXiv 2505.15960 (May 2025)
- **What:** Uses formal verification tools (Z3 + Isabelle) to automatically generate ground-truth
  step-level labels for process reward models, eliminating human annotation. Improves PRM
  generalization by 12-18% on out-of-distribution benchmarks vs PRMs trained on human-annotated data.
  Key insight: formal verification is distribution-agnostic — labels are provably correct regardless
  of the problem domain.
- **Relevance to Carnot:** Carnot's FoVer-labeled CoT corpus (57 pairs, Exp 442) was annotated by
  Z3-based VeriCoT. This paper confirms that Z3-labeled step data is higher quality than human-labeled
  data for training verification models. Implication: Carnot should EXPAND the FoVer corpus with
  additional Z3-labeled steps rather than collecting human labels. SC-Energy (Exp 944) trained on
  a larger Z3-labeled corpus should outperform the current AUROC=0.9017 checkpoint.
- **Concrete experiment:** Future milestone — SC-Energy v3 with expanded Z3-labeled corpus (500+
  step pairs via automated Z3 annotation). Expected: AUROC improvement from 0.9017 toward 0.95+.
- **When to incorporate:** Milestone 2026.04.77+ after .76 establishes Tier 2 wiring baseline.

### Thermodynamic Computing System for AI Applications
- **Paper:** Nature Communications, April 2025 (Coles et al.)
- **What:** Demonstrates physics-based stochastic processing units (SPU) built from RLC circuits
  performing Gaussian sampling and matrix inversion. SPUs integrate with FPGA for noise control
  and digital interface. Hardware achieves sampling at ~1 ns/sample vs ~1 ms for software MCMC,
  with ~100x lower power than GPU-based sampling.
- **Relevance to Carnot:** Directly validates the Extropic TSU (thermodynamic sampling unit) path.
  SPU architecture is essentially what Extropic Z1 targets: hardware Boltzmann sampling at
  nanosecond speeds. The Nature Comms paper provides peer-reviewed validation that thermodynamic
  computing is practical for constraint satisfaction, not just theoretical. Key for Phase 2
  roadmap: FPGA Ising (KV260) → SPU prototype → Extropic Z1 is the validated escalation path.
  SamplerBackend abstraction (REQ-KONA-006) should be tested with D-Wave as the next available
  non-CPU backend before Extropic ships.
- **When to incorporate:** Background reference for Phase 2/3 hardware roadmap. No new experiment
  needed — validates existing hardware trajectory.

## 2026-04-26 arxiv Scan (Milestone 2026.04.72 Planning)

### The Topological Trouble With Transformers
- **Paper:** arXiv 2604.17121 (Mozer, Siddiqui, Liu — April 2026)
- **What:** Argues that pure-feedforward transformers cannot reliably track dynamic state. As
  new inputs arrive, the evolving latent representation is pushed deeper into the layer
  stack — by the time a downstream consumer queries a shallow layer the relevant state has
  moved past it. The fix the authors prescribe is to shift to recurrent / continuous-thought
  architectures with implicit activation dynamics rather than explicit thought traces, and
  they offer a taxonomy of recurrent transformers along two axes (recurrence axis: depth vs
  step; input-to-recurrence ratio).
- **Relevance to Carnot — direct implications for live experiment lines:**
  - **Explains DRIFTProbe's repeated failure** (.69 not-viable → .70 marginal → .72 ensemble
    no-improvement). Carnot's existing DRIFTProbe attempts read state from a single hidden
    layer. Mozer's prediction is exactly that this can't work — the state isn't localized.
    A *depth-recurrent* probe that pools across layers (or attends over the layer stack)
    is the architecturally-aligned next attempt.
  - **Explains the math-IterativeSelfRepair zero-improvement** (Exp 930). Code repair
    succeeded (+72 pp HumanEval) because execution traceback is an *external* state-feedback
    signal that re-enters the model at the input layer. Math repair has no such external
    signal — each retry's diagnostic state is lost into deep layers. The +0 pp is not a
    bug, it's a topological prediction. The architecturally-aligned fix is an *external
    scratchpad* (re-feed prior-attempt errors as input text), not internal recurrence.
  - **Legitimizes Phase 3 (EBM/EBT foundation model).** Boltzmann/Gibbs/Ising sampler tiers
    are already recurrent — each MCMC step is a recurrence on the latent state. JEPA's
    iterative-encoding pattern likewise. Carnot's "continuous-latent-space, non-autoregressive,
    self-correcting" endgame sits exactly where Mozer prescribes.
  - **Supports KAN over MLP for Tier 4.** Exp 936's `real_data_improves_over_synthetic`
    shows KAN's spline representation is more robust than feedforward MLP. KAN's local
    spline structure doesn't have transformer's "push state deeper" topology.
- **Limitations the authors flag:** The taxonomy is theoretical; no concrete benchmarks or
  numerical comparisons in the abstract. Their characterisation of current solutions
  (dynamic depth, explicit thinking) as "computationally and memory inefficient" is
  qualitative.
- **Concrete experiments to consider for .73:**
  - **DRIFTProbe v3 — depth-recurrent**: read state from all layers via attention pooling
    (or a small RNN over the layer dimension), not single linear probe.
  - **MathIterativeSelfRepair v2 — external scratchpad**: feed prior-attempt error text
    back into the prompt for the next attempt, mirroring the code-repair traceback path.
- **When to incorporate:** Milestone 2026.04.73 — Phase 1 (probe restructure) and Phase 2
  (math-repair scratchpad). Required reading for any new probe / repair experiment.

### Symbolic-KAN: Discrete Symbolic Structure for KAN Interpretability
- **Paper:** arXiv 2603.23854 (April 2026)
- **What:** Augments KAN splines with discrete symbolic node labels (ADD, MUL, CMP, EQ) drawn from
  a predefined vocabulary. Each activation function is constrained to follow its symbolic label's
  behavior, making the learned constraint function interpretable: "node 3 checks addition, node 7
  checks comparison direction." Achieves 94% symbolic accuracy on arithmetic benchmark tasks.
- **Relevance to Carnot:** Carnot's KAN tier uses pure splines without symbolic structure. Symbolic
  labels would make the energy function interpretable and potentially more accurate on structured
  arithmetic constraints where the constraint types are known a priori.
- **Concrete experiment:** Exp 937 — Symbolic-KAN Constraint Verifier (Milestone 2026.04.72, Phase 4).
- **When to incorporate:** Milestone 2026.04.72 — Phase 4.

### SC-Energy: Set Consistency Energy Networks
- **Paper:** arXiv 2503.10695 (March 2026)
- **What:** Energy function that scores whether a SET of statements {s1,...,sn} is internally
  consistent. Contrastive training: E(coherent_set) << E(contradictory_set). Achieves AUROC=0.89
  on multi-statement consistency detection benchmarks. Permutation-invariant pooling enables
  variable-length statement sets.
- **Relevance to Carnot:** Carnot's global consistency checker (Exp 172, 100% detection) uses
  explicit logical rules. SC-Energy provides a learned alternative that generalizes across domains
  without hand-crafted rules. Complementary to per-step verification — catches contradictions
  spanning multiple statements.
- **Concrete experiment:** Exp 944 — SC-Energy Set Consistency v2. AUROC=0.9017 on GSM8K-derived
  coherent vs contradictory sets. Verdict: sc_energy_viable. First actual run of SC-Energy in
  this project (Exp 939 was blocked by gate-check failure, not algorithmic failure).
- **Status:** Implemented. SCEnergyModel in python/carnot/models/sc_energy.py.

### GRPO-VPS: Verifiable Process Supervision for Reasoning
- **Paper:** arXiv 2604.20659 (April 2026)
- **What:** Combines GRPO training with verifiable process supervision signals at the step level.
  Each reasoning step is scored for process quality, not just outcome correctness. Shows +4.2pp
  on GSM8K and +3.8pp on MATH over standard GRPO with outcome-only rewards.
- **Relevance to Carnot:** Carnot's R-PRM Tier 2.9 (Exp 924) found zero improvement in heuristic
  mode. GRPO-VPS suggests that step-level process supervision requires live model inference (not
  heuristics) to generate meaningful gradient signals. Use as reference for R-PRM redesign in .73.
- **When to incorporate:** Future milestone — R-PRM Tier 2.9 redesign with live model inference.

### EORM: Energy Outcome Reward Model for Lightweight Post-Hoc Verification
- **Paper:** arXiv 2505.14999 (May 2025)
- **What:** Uses a lightweight post-hoc energy function as an outcome reward model (ORM). The
  energy function is trained to assign low energy to correct chain-of-thought solutions and high
  energy to incorrect ones. Achieves 0.88 AUROC on GSM8K outcome verification with 8x fewer
  parameters than discriminative ORM baselines.
- **Relevance to Carnot:** Directly validates Carnot's energy-based verification approach. EORM's
  architecture is essentially Carnot's Ising/KAN stack applied as an ORM. The paper provides
  external validation of the approach and suggests that outcome-level energy scoring (not just
  step-level) is commercially viable. Reference for positioning Carnot vs standard ORM systems.
- **When to incorporate:** Background reference; no new experiment needed (already implemented).

### DebugRepair: Self-Directed Debugging for Program Repair
- **Paper:** arXiv 2604.19305 (April 2026)
- **What:** LLM-guided debugging where the model first generates a hypothesis about what is wrong,
  then applies a targeted repair based on the hypothesis. Achieves +8.2pp on HumanEval vs standard
  self-repair (arXiv 2604.10508 baseline). Hypothesis generation is the key: models that explain
  WHY the code is wrong repair it more effectively than models that directly retry.
- **Relevance to Carnot:** Exp 905 implemented basic iterative self-repair (execute-feedback-retry).
  DebugRepair suggests adding a "hypothesis about why wrong" step before retrying. For Carnot:
  after energy scoring identifies the worst attempt, generate "hypothesis: this response is wrong
  because [energy components indicate constraint violation]." Then use the hypothesis as a repair
  prompt. Expected: fewer retries needed per repair.
- **Concrete experiment:** Future milestone — DebugRepair integration with Carnot energy diagnosis.
- **When to incorporate:** Milestone 2026.04.73 — after Exp 930 establishes math repair baseline.

## 2026-04-26 arxiv Scan (Milestone 2026.04.71 Planning)

### R-PRM: Reasoning-Driven Process Reward Modeling
- **Paper:** arXiv 2503.21295 (March 2026)
- **What:** Improves process reward models by having the model reason over intermediate reasoning
  steps before producing a reward signal. Achieves +11.9 F1 points on ProcessBench and +8.5 on
  PRMBench vs discriminative PRM baselines. Key insight: reasoning over WHY a step is wrong
  provides stronger gradient signal than binary correct/incorrect labels.
- **Relevance to Carnot:** Carnot's SymCodeVerifier (Tier 2.5) and CausalReasoningVerifier
  (Tier 2.7) currently detect violations without explaining them. R-PRM's reasoning step can be
  adapted as Tier 2.9: a lightweight "why-wrong" reasoning probe between Tier 2.7 (causal) and
  Tier 3 (Ising). The reasoning output also provides repair hints for IterativeSelfRepair.
- **Concrete experiment:** Exp 924 — R-PRM Step Reward Tier 2.9: implement a reasoning-augmented
  step verifier that generates a brief "why" explanation before scoring. Measure AUC improvement
  vs direct scoring on GSM8K step labeling. Expected: AUC improvement from reasoning context.
- **When to incorporate:** Milestone 2026.04.71 — Phase 4 (Exp 924).

### Hierarchical Reward Models for Enhanced Reasoning
- **Paper:** arXiv 2503.13551 (March 2026)
- **What:** Hierarchical Reward Models (HRM) evaluate reasoning at two levels: (a) individual step
  quality and (b) consecutive step coherence. Outperforms flat PRMs on benchmarks where cascading
  errors (correct individual steps but wrong carry-forward) are common.
- **Relevance to Carnot:** HRM's two-level evaluation mirrors Carnot's cascade architecture.
  CausalReasoningVerifier (Tier 2.7) handles the carry-forward case. HRM validates that
  hierarchical evaluation is worth the architectural complexity. The "step coherence" metric
  aligns with Carnot's global consistency checker approach.
- **When to incorporate:** Reference for Tier 2.7 enhancement and Tier 2.9 design decisions.

## 2026-04-26 arxiv Scan (Milestone 2026.04.70 Planning)

### Iterative Self-Repair in LLM Code Generation
- **Paper:** arXiv 2604.10508 (April 2026)
- **What:** Investigates iterative self-repair (feeding execution errors back to the LLM for
  correction) across 7 models (Llama 4, Qwen3 32B, Gemini 2.5) on HumanEval and MBPP. With up
  to 5 repair attempts, self-repair universally improves pass rates: +4.9 to +17.1 pp on
  HumanEval, +16.0 to +30.0 pp on MBPP. Most gains concentrate in the first 2 rounds. Assertion
  errors (logical mistakes) are hardest to repair (~45%); syntax and name errors repair at much
  higher rates.
- **Relevance to Carnot:** Carnot's code repair has been blocked for 12+ consecutive milestones
  because ArithmeticExtractor extracts zero constraints from Gemma4/Qwen3 IT responses. Iterative
  self-repair bypasses constraint extraction entirely: run the generated code, capture the execution
  error, and feed it back to the LLM for a second attempt. No ArithmeticExtractor needed. This is
  the path to first positive HumanEval improvement. Key insight: Carnot's energy-based verification
  can SELECT which attempt to keep (lowest energy score = best candidate), improving over naive
  self-repair by choosing the best of N attempts, not just the last.
- **Concrete experiment:** Exp 905 — IterativeSelfRepair v1: run Gemma4-E4B-it on 25 HumanEval
  problems, 3 attempts each. Execute code, feed errors back to model. Use Carnot's VerifyRepairPipeline
  to select the lowest-energy attempt. Target: signed_improvement > 0 (first in 12+ milestones).
- **When to incorporate:** Milestone 2026.04.70 — Phase 1 (Exp 905).

### LLMloop: Automated Iterative Feedback Loops for Code Generation
- **Paper:** arXiv 2603.23613 (March 2026)
- **What:** Framework to automatically improve LLM-generated code through compilation and execution
  feedback loops. First loop ensures generated code compiles; second loop runs tests and feeds
  failure traces back to the LLM. Shows consistent improvement on Java code generation benchmarks.
- **Relevance to Carnot:** Complements arXiv 2604.10508 (iterative self-repair). The two-stage
  feedback loop (compile first, test second) is directly applicable to HumanEval where Python
  syntax errors and runtime errors require different remediation paths. Carnot can implement:
  Stage 1 = AST syntax check (already have ConstrainedDecodingPreFilter), Stage 2 = execute test
  cases + Carnot energy scoring to rank repairs.
- **Concrete experiment:** Incorporated into Exp 905 and Exp 906 as the feedback loop structure.
- **When to incorporate:** Milestone 2026.04.70 — folded into Exp 905/906.

### AutoKnots: Adaptive Knot Allocation for Spline Interpolation
- **Paper:** arXiv 2412.13423 (December 2025)
- **What:** Proposes adaptive knot placement where knot locations are trained jointly with spline
  parameters. Free-knot adaptation consistently reduces errors vs fixed grids. KAN 2 already
  supports post-training grid refinement (nested spline spaces preserve training progress when
  knots are added). Key insight: high-density activation regions benefit from finer knot spacing;
  low-density regions waste parameters with fine grids.
- **Relevance to Carnot:** Directly informs FR-11 Tier 4 KAN adaptive structure (Exp 910).
  The "nested spline spaces" property means we can add knots to high-activation splines without
  retraining from scratch — KAN 2-style grid refinement preserves existing learned parameters.
  This makes Tier 4 restructuring cheap: add capacity where needed, keep everything else unchanged.
- **Concrete experiment:** Exp 910 — FR-11 Tier 4 KAN Seed using AutoKnots-style adaptive
  refinement: run forward pass to build activation histograms, then add knots to the top-30%
  high-activation splines using grid refinement (not full retraining).
- **When to incorporate:** Milestone 2026.04.70 — Phase 3 (Exp 910).

### Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling
- **Paper:** arXiv 2604.13386 (April 2026)
- **What:** Shows that multi-layer probe ensembling consistently outperforms single-layer probes
  for detecting internal model properties. Ensembling probes across layers (L4, L8, L12, L16)
  via learned weighting improves AUROC by 3-8% vs best single layer. The weights adapt to
  where each model stores the relevant information.
- **Relevance to Carnot:** DRIFT probe (Exp 911) uses hidden states at 4 fixed layers. Changing
  from per-layer cosine similarity to a learned ensemble of the 4-layer drift signatures should
  improve probe_auc. The paper validates that multi-layer ensembling is worthwhile even for
  lightweight linear probes. Implemented by replacing single LogisticRegression(X) where X is
  the 3-value drift vector with LogisticRegression(X) where X = learned_weighted_ensemble(per_layer_drifts).
- **Concrete experiment:** Incorporated into Exp 911 (DRIFT Multi-Layer Ensemble Probe).
- **When to incorporate:** Milestone 2026.04.70 — Phase 4 (Exp 911).

### Var-JEPA: Bridging Predictive and Generative Self-Supervised Learning
- **Paper:** arXiv 2603.20111 (March 2026)
- **What:** Formalises the JEPA framework as a deterministic latent variable model with a
  corresponding ELBO objective. Shows JEPA and VAE are complementary: JEPA is a deterministic
  encoder with implicit predictive prior; VAE adds explicit distributional uncertainty. Var-JEPA
  combines both: the ELBO minimises reconstruction + KL(q(z|x) || p(z|context)), identical to
  VJEPA's objective but derived from first principles.
- **Relevance to Carnot:** Theoretical confirmation that Carnot's VariationalJEPAPredictor
  (ood_auc=0.9211) is on the right architectural track. Var-JEPA's ELBO formulation provides
  a cleaner gradient signal for SVAMP-specific retraining (Exp 908): the reconstruction term
  forces the model to explain SVAMP response structure, while the KL term prevents OOD collapse.
- **When to incorporate:** Theoretical basis for Exp 908 SVAMP VJEPA v3 training.

## 2026-04-25 arxiv Scan (Milestone 2026.04.67 Planning)

### VJEPA: Variational Joint Embedding Predictive Architecture as Probabilistic World Model
- **Paper:** arXiv 2601.14354 (January 2026)
- **What:** Extends JEPA with a variational objective: learns predictive distributions over future
  latent states via L = E[log p(z_t | c_{t-1})] - KL[q(z_t | x_t) || p(z_t | c_{t-1})].
  Provides collapse-avoidance guarantees via KL regularization. Outperforms standard JEPA on
  multi-step prediction tasks by modeling uncertainty explicitly.
- **Relevance to Carnot:** Carnot's JEPA predictor (Tier 3) currently collapses to AUC<0.5 under
  OOD conditions because it makes point predictions without uncertainty estimates. VJEPA's
  variational objective directly addresses this: the KL term prevents representation collapse,
  and uncertainty estimates indicate when the predictor should defer to full Ising verification
  (high uncertainty = run Tier 3, low uncertainty = skip). This is Tier 3's key missing piece.
- **Concrete experiment:** Exp 877 — VariationalJEPAPredictor: replace JEPA's deterministic
  predictor with a variational one (encoder q, prior p, reparameterization trick). Train on
  FoVer corpus. Compare OOD AUC: JEPA v25 (deterministic) vs VJEPA (variational).
  Target: OOD AUC > 0.65. Hardware path: variational encoder is a small MLP — NPU-compatible.
- **When to incorporate:** Milestone 2026.04.67 — Phase 4 (Exp 877).

### Efficient Optimization Accelerator Framework for Multistate Ising Problems on FPGA
- **Paper:** arXiv 2505.20250 (May 2025)
- **What:** Implements 1,024-neuron all-to-all connected probabilistic Ising machine on Xilinx
  FPGA. Achieves 10,000x speedup vs GPU for Ising constraint satisfaction. Key technique:
  vectorized p-bit update with compressed adjacency (sparse storage + BRAM lookup). Reduces
  physical neurons 1.5-4x via variable elimination before synthesis.
- **Relevance to Carnot:** iCE40 N=8 synthesis fits at 134 LUTs (Exp 859). The variable
  elimination technique could allow N=16 to fit where direct synthesis overflows (12258 LCs
  in Exp 851). Sparse adjacency storage is directly applicable to Carnot's constraint graphs,
  which are typically sparse (2-3 constraints per variable). The 10,000x speedup validates
  the FPGA path for Tier 1-4 hardware acceleration.
- **Concrete experiment:** Enhancement to Exp 876 (iCE40 inertia v2) — add sparse adjacency
  storage: instead of full N×N coupling matrix, store (i, j, J_ij) triples in BRAM. This
  enables N=16 synthesis without the register expansion that caused RETRO-ICE40-N16.
- **When to incorporate:** Milestone 2026.04.67 — Phase 4 (Exp 876), also gates N=16 retry.

### Correctness-Guaranteed Code Generation via Constrained Decoding
- **Paper:** arXiv 2508.15866 (August 2025)
- **What:** Uses a context-sensitive parser to constrain LLM token generation via dynamic regex
  trees. At each generation step, the parser computes the set of valid next tokens given the
  partial AST, then masks invalid tokens in the LLM's logit vector. Achieves 100% syntactic
  correctness on Python output at minimal (<5%) latency overhead.
- **Relevance to Carnot:** Carnot's CodeExtractor + VerifyRepairPipeline corrects code AFTER
  generation. This paper provides a complementary approach: constrain DURING generation so
  violations never appear. Combining both — constrained generation + energy-based post-hoc
  verification — is stronger than either alone. The constrained decoder could reduce
  ArithmeticExtractor FP rate by preventing clearly-invalid intermediate steps from
  appearing in the CoT.
- **Concrete experiment:** Future milestone — ConstrainedDecodingIntegration: wire the
  paper's constrained decoding approach as a pre-filter before VerifyRepairPipeline for
  HumanEval code generation. Measure: FP rate reduction + pass@1 vs unconstrained.
- **When to incorporate:** Milestone 2026.04.68 (after Exp 870 gives live code repair baseline).

### Neural Probe-Based Hallucination Detection for LLMs
- **Paper:** arXiv 2512.20949 (December 2025)
- **What:** Lightweight MLP probes on frozen LLM hidden states identify hallucinated tokens
  at inference time. Probes are 2-layer MLPs trained on (hidden_state, hallucination_label)
  pairs. Achieves F1 > 0.85 with <0.1ms overhead per token on Qwen-family models.
- **Relevance to Carnot:** NUP Probe v4 (Exp 523, AUC=1.0) uses bigram contrastive probes —
  this paper validates that small MLP probes on actual hidden states are competitive. The
  0.1ms latency makes it feasible as a Tier 0 pre-filter on the verification cascade.
  Critically, 2-layer MLPs are NPU-native (GEMM + ReLU), which means the AMD XDNA NPU
  could run this probe while the CPU runs Ising verification in parallel.
- **Concrete experiment:** Future milestone — HiddenStateHalluProbe: train 2-layer MLP probe
  on Qwen3.5-0.8B final-layer hidden states labeled by FoVer. Compare with NUP Probe v4.
  If AUC is competitive, replace or augment Tier 0c with MLP probe. Target: AUC > 0.90.
- **When to incorporate:** Milestone 2026.04.68 or later (needs labeled hidden state corpus).

### Process Reward Models That Think (ThinkPRM)
- **Paper:** arXiv 2504.16828 (April 2026)
- **What:** Generative process reward model that produces chain-of-thought verification before
  scoring. "Think first, then judge" approach. Outperforms discriminative PRMs by 8% on
  GPQA-physics OOD. The generative prior provides regularization that prevents overfitting
  to in-distribution scoring patterns.
- **Relevance to Carnot:** ThinkPRM is already in Carnot's Tier 0a (CarnotThinkProbe, Exp 444).
  Key finding: the THINKING step is the source of OOD generalization, not just the final score.
  Applying this to JEPA v25: add a reasoning step ("why is this CoT step wrong?") before
  the energy score. This is the variational analog — the reasoning IS the prior p(z_t | c_{t-1}).
  Direct connection to VJEPA: the "thought" is the predicted latent.
- **When to incorporate:** Incorporated into VJEPA experiment design (Exp 877) as theoretical
  basis for the prior distribution.

## 2026-04-25 arxiv Scan (Milestone 2026.04.66 Planning)

### HalluSAE: Detecting Hallucinations via Sparse Auto-Encoder Geometry
- **Paper:** arXiv 2604.16430 (April 2026)
- **What:** Uses Sparse Auto-Encoders (SAE) applied to LLM residual stream activations to extract
  interpretable features. Tracks temporal evolution of SAE feature geometry across reasoning steps —
  hallucinated reasoning shows systematically different feature dynamics (higher dimensional "energy"
  in SAE feature space) compared to correct reasoning. First work using dynamic SAE geometry for
  general factual hallucination detection. Requires no labels at inference time.
- **Relevance to Carnot:** The "geometric potential energy" framing maps directly onto Carnot's
  energy tier cascade. SAE features are already activations; the geometric energy is a natural
  Tier 0j advisory signal alongside HalluField (Tier 0e) and SemanticEnergyProbe (Tier 0f).
  Key advantage: SAE geometry is orthogonal to all existing tiers (it operates on feature
  geometry, not logit distributions or sentence semantics). No GPU needed at inference time
  if SAE dictionary is pre-computed — just a dot-product in sparse feature space.
- **Concrete experiment:** Exp 863 — HalluSAEGeometricProbe: implement a lightweight SAE geometry
  energy using pre-computed bigram feature dictionary (like NUP Probe v4 in Exp 523). Compute
  temporal feature-energy trajectory across CoT steps. Compare AUC on 50 synthetic CoT pairs.
- **When to incorporate:** Milestone 2026.04.66 — Phase 3 new probes (Exp 863).

### Fully Parallel Densely Connected Probabilistic Ising Machine with Inertia
- **Paper:** arXiv 2604.17109 (April 2026)
- **What:** Implements a fully parallel, synchronous-update probabilistic Ising machine (PIMI) on
  FPGA with an inertia term that modifies spin dynamics: each spin tracks an exponential moving
  average (EMA) of its local field, damping oscillations and reducing sweep count by 15-25x vs
  standard Gibbs. Hardware-software co-design achieves real-time constraint satisfaction with
  much fewer iterations. The inertia term enables fully parallel (non-checkerboard) updates.
- **Relevance to Carnot:** This is directly relevant to two open RETROs: RETRO-ISING-INJECTION-NO-
  DISCRIMINATION (energy delta identical for error/clean code — insufficient mixing) and
  RETRO-ICE40-N16-UNEXPECTED-EXPANSION (N=16 expanded from 2 LUTs to 12258 LCs at P&R, likely
  from registered spin state). The inertia approach (EMA tracking per-spin) is lighter hardware
  than registered spin states: EMA can be implemented with only multipliers + adders, reducing
  LUT count substantially vs full flip-flop registers. Additionally, inertia significantly
  improves discrimination between close-energy configurations (exactly the injection problem).
- **Concrete experiment:** Exp 860 — InertiaIsingSamplerBenchmark: implement Python simulation of
  PIMI with inertia (EMA alpha=0.5) vs standard checkerboard Gibbs. Compare: discrimination_delta
  (energy gap between correct/incorrect constraint configurations) and mixing_sweeps_to_converge.
  Target: inertia reduces mixing sweeps by 5x+ and improves discrimination. CPU simulation.
- **When to incorporate:** Milestone 2026.04.66 — Phase 2 (Exp 860).

### Self-Adaptive Ising Machines for Constrained Optimization
- **Paper:** arXiv 2501.04971 (January 2026)
- **What:** Introduces self-adaptive Ising machines that iteratively reshape their energy landscape
  using Lagrange relaxation of constraints. When a constraint is violated, the Lagrange penalty is
  automatically increased, making the energy landscape "steeper" around the constraint boundary.
  This eliminates the need for manual tuning of constraint weights and enables the Ising machine
  to self-organize around the active constraint boundary.
- **Relevance to Carnot:** EmbeddingConstraintStore retrieval produces zero energy delta even
  after L2-normalization fix (RETRO-CONSTRAINT-ZERO-DELTA may persist in .66 if Exp 847 partial).
  Self-adaptive Lagrange Ising directly addresses this: instead of static constraint injection,
  adaptively increase the Lagrange weight for constraints that are repeatedly violated. This is
  the missing link between Tier 1 online tracking (which detects violations) and IsingEBM energy
  computation (which currently uses fixed coupling weights). The adaptation loop IS the self-
  learning Tier 1 mechanism designed in research-program.md.
- **Concrete experiment:** Exp 862 — LagrangeAdaptiveIsingConstraints: implement Lagrange-adaptive
  coupling update in IsingEBM. When EmbeddingConstraintStore retrieves a constraint and violation
  is detected, increase the coupling weight for that constraint type by factor(1 + lr * violation_count).
  Run 5-session relay. Compare delta_s1_to_s5 vs non-adaptive baseline.
- **When to incorporate:** Milestone 2026.04.66 — Phase 3 self-learning (Exp 862).

### Streaming Hallucination Detection in Long Chain-of-Thought Reasoning
- **Paper:** arXiv 2601.02170 (January 2026)
- **What:** Treats hallucination in long CoT as an evolving latent state, not a static binary label.
  Introduces prefix-level cumulative hallucination signal (PHaS) that tracks global evolution over
  the entire trajectory — step-level judgments are local observations fed into a running state
  estimate. Streaming detector significantly outperforms end-of-response or per-step detection
  because it exploits trajectory dynamics (hallucination tends to compound once it starts).
- **Relevance to Carnot:** All Carnot cascade tiers currently evaluate the FINAL response, not
  the reasoning trajectory. For Gemma4 and Qwen3 with chain-of-thought reasoning, the hallucination
  may start at step 3 and compound through step 7. A streaming detector that tracks PHaS across
  CoT steps would catch compounding errors earlier and integrate naturally with Tier 3.5 (JEPA
  predictive verification). Maps to Tier 0i as a new advisory signal.
- **Concrete experiment:** Exp 861 — StreamingCoTHalluDetector (Tier 0i): implement prefix-level
  cumulative hallucination signal from per-step EORM scores. At each CoT step, update running state
  estimate: phas_t = alpha * eorm_score_t + (1-alpha) * phas_{t-1}. Flag is_streaming_unstable
  if phas exceeds threshold at any step. Add to VerificationCertificate as advisory.
- **When to incorporate:** Milestone 2026.04.66 — Phase 3 new probes (Exp 861).

### Memory Bank Compression for Continual Adaptation of Large Language Models
- **Paper:** arXiv 2601.00756 (January 2026)
- **What:** Compresses information from new data into a memory bank, aggregating document memories
  to answer queries while keeping LLM parameters frozen. Key technique: memory compression via
  learned attention over stored embeddings. Prevents catastrophic forgetting while enabling
  online adaptation at inference time.
- **Relevance to Carnot:** EmbeddingConstraintStore grows unboundedly as sessions accumulate.
  Exp 748 showed a plateau at session 2 (templates added in S1, then replay-only). Memory bank
  compression addresses this: compress the accumulated constraint embeddings into a smaller bank
  that retains the most discriminative constraint patterns. This is the missing infrastructure
  for sustained Tier 2 learning across many sessions.
- **Concrete experiment:** Exp 865 — ConstraintMemoryBankCompression: implement compressed
  memory bank (K=32 centroid embeddings) using kmeans-style online clustering of EmbeddingConstraintStore.
  Each new constraint is assigned to the closest centroid or starts a new cluster if distance > threshold.
  Compare: retrieval AUROC with raw store vs compressed bank after 10 sessions.
- **When to incorporate:** Milestone 2026.04.66 — Phase 4 (Exp 865).

### Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks
- **Paper:** arXiv 2604.03345 (April 2026)
- **What:** Analyzes the hardware inference complexity of KANs systematically: spline evaluation cost,
  activation memory footprint, parallelism opportunities. Shows that KANs with piecewise-linear splines
  (linear interpolation between knots) are highly FPGA-efficient: each spline evaluation requires
  only 2 multiplications and 1 addition, with no non-linear functions (no exp/tanh hardware needed).
  Provides LUT count estimates for KAN layers on Xilinx and Lattice FPGAs.
- **Relevance to Carnot:** Carnot's KAN energy tier (carnot-kan) is the candidate for FPGA acceleration
  after the Ising sampler. The paper's LUT estimates show that a KAN with 8 knots per spline and
  64 hidden units would require ~2,000 LUTs on iCE40 HX8K (within budget). This directly informs
  the KV260 synthesis roadmap: implement Ising first (current work), then KAN energy tier (next hardware
  target). The piecewise-linear variant is compatible with Carnot's KAEMEnergy model.
- **Concrete experiment:** Exp 866 — KANHardwareComplexityAnalysis: measure per-knot LUT estimates
  for Carnot's UnivariateKAEMLayer (8 knots, piecewise-linear). Simulate the iCE40 synthesis LUT count.
  Compare vs Ising N=16 (Exp 859). Determine which tier to synthesize for KV260 next.
- **When to incorporate:** Milestone 2026.04.66 — Phase 4 (Exp 866).

### Digitally Optimized Initializations for Fast Thermodynamic Computing
- **Paper:** arXiv 2603.24183 (March 2026)
- **What:** Proposes Mpemba-optimized initializations computed digitally, then encoded into
  thermodynamic hardware to suppress slow relaxation modes. Key insight: certain initialization
  states can dramatically reduce the thermalization time by exploiting the spectral gap of the
  Ising Hamiltonian. Digital precomputation + thermodynamic relaxation is hybrid — the digital
  precomputation selects the initialization, then hardware relaxes from it rapidly.
- **Relevance to Carnot:** The Gibbs warm-start fix (Exp 846, warm_start_sweeps=500) worked but
  is expensive. Mpemba initialization could reduce burn-in from 500 to ~50 sweeps by computing
  an optimal starting state from the bias vector h_i alone (pure CPU precomputation). This
  directly improves the arbiter (faster energy measurement) and the Ising injection pipeline.
- **Concrete experiment:** Enhancement to Exp 860 (InertiaIsingSampler) — add Mpemba initialization:
  compute optimal starting magnetization from spectral properties of J+h, use as initial spin state
  instead of random ±1. Compare burn-in sweeps to convergence.
- **When to incorporate:** Milestone 2026.04.66 — integrated into Exp 860 inertia benchmark.

## 2026-04-25 arxiv Scan (Milestone 2026.04.64 Planning)

### Dynamic and Generalizable Process Reward Modeling (DG-PRM)
- **Paper:** arXiv 2507.17849 (July 2025)
- **What:** Proposes DG-PRM that dynamically selects relevant and effective rewards for each
  domain at inference time. Achieves strong OOD generalization by learning a domain-reweighting
  function that adapts PRM scores based on the input distribution shift. Standard PRMs show
  significant OOD performance drops; DG-PRM recovers 5-8% AUC across held-out domains.
- **Relevance to Carnot:** JEPA v23 collapsed to AUC=0.04 on ARC (Exp 825) despite AUC=0.81
  on training eval. Root cause: no ARC training data AND no domain adaptation at inference.
  DG-PRM's domain-reweighting approach directly addresses this: train a lightweight domain
  classifier alongside JEPA, multiply JEPA output by domain weight at inference. Expected
  to recover ARC from 0.04 to > 0.50 without adding domain-specific training data.
- **Concrete experiment:** Exp 834 — JEPA v24 with DG-PRM domain reweighting. Balance corpus
  20+20+20+10 (GSM8K/HumanEval/ARC/SVAMP). Add domain-reweighting head trained jointly.
- **When to incorporate:** Milestone 2026.04.64 — Phase 2 JEPA fix.

### DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning
- **Paper:** arXiv 2505.20241 (May 2025)
- **What:** Addresses distribution shift in PRM training via per-domain importance weights.
  Quality imbalance across reasoning domains is the root cause of OOD failure. DreamPRM
  computes per-domain loss weights proportional to validation error, ensuring rare domains
  (like planning/ARC) get upweighted during training even when underrepresented in corpus.
- **Relevance to Carnot:** Complementary to DG-PRM. DreamPRM's per-domain loss weighting
  during training directly addresses the GSM8K-vs-ARC imbalance problem (Exp 825: 0.36 vs 0.04).
  Concrete fix: compute per-domain validation loss before training, set ARC weight = 5x,
  GSM8K weight = 1x. Prevents the model from ignoring ARC examples during optimization.
- **Concrete experiment:** Integrated into Exp 834 as augmentation to DG-PRM.
- **When to incorporate:** Milestone 2026.04.64 — Phase 2 JEPA fix.

### ΔEnergy: Energy Change for OOD Detection and Generalization
- **Paper:** arXiv 2510.11296 (October 2025)
- **What:** Uses energy score *changes* (delta between positive and negative samples) rather
  than raw energy scores to distinguish in-distribution from out-of-distribution data.
  Key insight: absolute energy is unreliable OOD; energy delta is distribution-invariant.
  Achieves improvements in both OOD detection and OOD generalization in vision-language models.
- **Relevance to Carnot:** JEPA is trained on raw energy scores; the delta approach would train
  on (E_correct - E_incorrect) gaps instead. This is distribution-invariant: a model trained
  on GSM8K energy deltas will generalize to ARC energy deltas because the relative gap
  is preserved even when absolute scales differ. Maps to the triplet loss in Exp 824 —
  enhance the loss to weight by energy-delta magnitude.
- **Concrete experiment:** Exp 834 — add ΔEnergy loss weighting: triplets with larger
  (E_negative - E_positive) gaps get higher loss weight, forcing the model to focus on
  clearly discriminative examples. This complements DG-PRM domain reweighting.
- **When to incorporate:** Milestone 2026.04.64 — Phase 2 JEPA fix (Exp 834).

### Schema-Constrained Generation for Agent Memory
- **Paper:** arXiv 2604.20117 (April 2026)
- **What:** Uses constrained LLM decoding via Trie-based structure to ensure agents generate
  valid memory retrieval keys. Mathematically precludes structural hallucinations in memory
  access by constraining the generation beam to valid schema paths. Achieves 100% structural
  correctness with minimal latency overhead.
- **Relevance to Carnot:** EmbeddingConstraintStore delta=0 in Exp 821 despite injection fix.
  Root cause hypothesis: constraints being stored have inconsistent schema (free-form strings
  not aligned to violation taxonomy). Schema-Constrained approach: define a fixed schema for
  constraint storage (violation_type, affected_step, constraint_expression), enforce at write
  time. This would prevent schema drift that may explain why retrieved constraints don't
  produce energy deltas.
- **Concrete experiment:** Exp 836 — add schema validation to EmbeddingConstraintStore.write():
  constraints must conform to (violation_type, step_id, expression) schema. Run diagnostic
  experiment to see if schema enforcement increases retrieval precision.
- **When to incorporate:** Milestone 2026.04.64 — Phase 2 constraint fix (Exp 836).

## 2026-04-24 arxiv Scan (Milestone 2026.04.62 Planning)

### Retrieval-Augmented Process Reward Model for OOD Generalization
- **Paper:** arXiv 2502.14361 (February 2026)
- **What:** Addresses OOD generalization in process reward models via retrieval augmentation.
  Identifies two distinct failure modes: step-OOD (reasoning step type unseen in training) and
  question-OOD (problem domain unseen). Retrieval-augmented PRM retrieves similar training steps
  and uses their labels as soft supervision, achieving 6-8% OOD improvement on MATH-500 variants.
- **Relevance to Carnot:** JEPA v21 OOD AUC=0.2444 (all-time low) was caused by a wiring miss
  (not an algorithmic failure), but JEPA has also failed OOD in 8 prior retrains with correct
  wiring. RA-PRM's retrieval augmentation directly addresses the OOD problem: for each training
  step, retrieve K similar steps from the training corpus (via EmbeddingConstraintStore) and
  include their FoVer labels as soft targets. This provides step-level distribution shift
  correction without requiring more training data.
- **Concrete experiment:** Exp 809 — if JEPA v22 CPMI-wired retrain still fails OOD after
  the wiring fix, apply RA-PRM: use EmbeddingConstraintStore to retrieve 3 similar FoVer-labeled
  steps per training example and average their labels as soft targets. Compare OOD AUC:
  JEPA v22 (wiring fix only) vs JEPA v22 + RA-PRM (retrieval augmentation).
- **When to incorporate:** Milestone 2026.04.62 — Phase 1 JEPA OOD enhancement (Exp 809).

### Variable Granularity Search for Test-Time Compute Scheduling
- **Paper:** arXiv 2505.11730 (May 2025)
- **What:** Introduces Variable Granularity Search (VG-Search), showing that verification
  frequency (how often to check a reasoning trace) trades off accuracy vs. compute cost.
  Unifies beam search (per-token) and Best-of-N (end-only) as extremes on a granularity
  spectrum. Optimal granularity depends on question difficulty; adaptive scheduling outperforms
  fixed-frequency by 3-5% at equal compute budgets.
- **Relevance to Carnot:** Carnot's cascade (Tier 0 → Tier 3) already schedules energy
  computation adaptively, but the granularity decision is fixed (always full-response
  verification at each tier). VG-Search provides a principled framework to decide WHEN to
  invoke constraint checking during generation: high-uncertainty intermediate states get
  frequent checks (more Ising calls), low-uncertainty states get MARS-style fast-path skip.
  This is the next evolution beyond the static MARS margin gate implemented in Exp 796.
- **Concrete experiment:** Exp 815 — VGSearchCarnot: implement variable-granularity energy
  scheduling in ThreeTierPipeline. At each tier, track running uncertainty estimate
  (energy variance across last 3 checks). If variance is low, extend interval before next
  check. Measure: constraint_calls_reduced vs pass@1 degradation on 50 GSM8K questions.
  Target: 30% fewer Ising calls at equal accuracy. CPU-only (no GPU needed for energy tiers).
- **When to incorporate:** Milestone 2026.04.62 — Phase 4 compute efficiency (Exp 815).

### Capacity-Constrained Continual Learning
- **Paper:** arXiv 2507.21479 (July 2025)
- **What:** Formalizes optimal resource allocation for continual learning under memory and
  compute constraints. Derives closed-form solutions for which model components to update
  given a fixed per-update budget. Key result: selective parameter update (targeting only
  high-uncertainty parameters) outperforms full-parameter update by 7-12% under tight budgets.
- **Relevance to Carnot:** Self-learning Tier 1 (online weight updates) currently updates ALL
  constraint weights uniformly after each verification. This is wasteful when most constraints
  are well-calibrated. Capacity-constrained optimization provides the rule: update only
  constraints with high energy variance (poorly calibrated), freeze well-calibrated ones.
  This maps directly to EmbeddingConstraintStore: when storing a new constraint vector,
  only apply orthogonality update to constraints with retrieval uncertainty > threshold.
- **Concrete experiment:** Enhancement to Exp 813 (FR-11 Tier 1 Live Relay) — apply
  capacity-constrained update rule: compute energy variance per constraint type over the
  last session; update only top-K highest-variance constraints. Compare: full update
  vs capacity-constrained update — does selective update improve precision monotonicity?
- **When to incorporate:** Milestone 2026.04.62 — augmentation to FR-11 relay experiment.

### Jailbreaking Leaves a Trace: Detecting Jailbreaks via Internal Representations
- **Paper:** arXiv 2602.11495 (February 2026)
- **What:** Demonstrates that successful jailbreaks produce distinctive patterns in LLM
  internal representations (activation space) that persist across model layers. A lightweight
  linear probe trained on intermediate activations achieves 95.8% detection accuracy with
  <1ms overhead per request. Detection works zero-shot on unseen jailbreak types.
- **Relevance to Carnot:** JailbreakDetectionKAN (Tier 0h, Exp 775, AUC=1.0) was built as a
  KAN-based safety gate. This paper suggests an alternative or complementary path: activation-
  space probing is even lighter (linear probe vs KAN) and generalizes better zero-shot. The
  activation probe maps to Carnot's SpilledEnergyDetector architecture — both use intermediate
  representations as energy signals. Could replace or supplement Tier 0h.
- **Concrete experiment:** Future milestone — ActivationJailbreakProbe: implement linear probe
  on Qwen3.5-0.8B intermediate layers. Compare with JailbreakDetectionKAN on 50 jailbreak
  prompts and 50 benign prompts. Report: AUC, latency, generalization to unseen attack types.
- **When to incorporate:** Milestone 2026.04.63 — Safety products track.

### OSS-CAD-Suite: Pre-Built Open-Source FPGA Toolchain
- **Reference:** YosysHQ/oss-cad-suite-build (GitHub)
- **What:** Pre-built binary distribution of yosys, nextpnr, icestorm, and related tools
  for Linux (x86_64 and aarch64). No build required — download tarball, extract, set PATH.
  Updated weekly with latest upstream versions. Includes yosys, nextpnr-ice40, nextpnr-ecp5,
  nextpnr-xilinx, icepack, icetime, and 20+ supporting tools.
- **Relevance to Carnot:** RETRO-KV260-TOOLS-UNAVAILABLE has failed for 3 consecutive
  milestones (Exps 791, 794, 804) because `sudo pacman -S yosys nextpnr icestorm` does not
  work in the conductor environment. OSS-CAD-Suite bypasses the package manager entirely:
  download the release tarball from GitHub, extract to ~/tools/oss-cad-suite/, and prepend
  ~/tools/oss-cad-suite/bin to PATH. No sudo required. This is the correct next attempt.
- **Concrete experiment:** Exp 807 — OSSCADInstall: download oss-cad-suite-build release
  tarball (latest release from GitHub API), extract to ~/tools/oss-cad-suite/, verify
  `yosys --version`, `nextpnr-ice40 --version`, `icepack --help` all pass. Run minimal
  2-spin Ising synthesis as proof-of-concept. Gates Exp 816 (KV260 synthesis v2).
- **When to incorporate:** Milestone 2026.04.62 — Phase 0 FPGA unblock (Exp 807).

## 2026-04-24 arxiv Scan (Milestone 2026.04.61 Planning)

### Semantic Interference in Neural Memory: Orthogonality Constraint for Constraint Retrieval
- **Paper:** arXiv 2601.15313 (January 2026)
- **What:** Identifies Semantic Interference — as semantically similar facts accumulate in embedding-based
  memory, embeddings collapse and retrieval accuracy degrades to near-random. Demonstrates that structured
  SPO (Subject-Predicate-Object) format achieves 11x higher recall than unstructured text embeddings.
  Proposes orthogonality regularization during encoding to maintain separation between concepts.
- **Relevance to Carnot:** Directly explains RETRO-CONSTRAINT-ZERO-DELTA from Exp 788: scalar keyword-count
  encoding of carry/sign/unit error patterns semantically interfere with each other, producing delta=0.0.
  The fix is to encode constraints as SPO tuples (e.g., {subject: "arithmetic_step_3", predicate: "violates",
  object: "carry_propagation"}) with sentence-transformer embeddings + orthogonality regularization.
  This replaces keyword_count integers with discriminative constraint vectors that don't collapse.
- **Concrete experiment:** Exp 800 — EmbeddingConstraintStore: implement SPO-format constraint
  encoding using sentence-transformers + orthogonality regularization. Replace scalar keyword
  counts in CaseMemoryTemplateWiring. Test: retrieval AUC on 5 constraint types. Target delta > 0.
- **When to incorporate:** Milestone 2026.04.61 — Phase 3 embedding constraint retrieval (Exp 800).

### CPMI: Contrastive Pointwise Mutual Information for Efficient Process Reward Modeling
- **Paper:** arXiv 2604.10660 (April 2026)
- **What:** Proposes Contrastive Pointwise Mutual Information (CPMI) as an automatic reward labeling
  method. For each reasoning step, computes how much the step increases mutual information relative to
  hard negative alternatives sampled from the model distribution. Provides step-level supervision without
  human annotation. Used by Exp 576-class JEPA CPMI Pair Builder.
- **Relevance to Carnot:** JEPA v20 (ood_auc=0.4467) fails OOD because training pairs are insufficiently
  contrastive — easy correct/incorrect pairs don't teach generalization. CPMI hard negatives force the
  predictor to distinguish subtly-wrong from subtly-right steps. Applied to JEPA v21: augment the
  FoVer v2 corpus with CPMI-ranked contrastive pairs where the negative is a plausible wrong step,
  not a random wrong step. This is the key change to break the 8-consecutive-retrain failure pattern.
- **Concrete experiment:** Exp 798 — CPMIContrastivePairBuilder: compute CPMI scores for each FoVer
  step pair, select top-K hard negatives (CPMI score in 0.2-0.6 range), augment training corpus.
  Compare JEPA v21 OOD AUC: standard pairs vs CPMI-augmented. Target: OOD AUC >= 0.75.
- **When to incorporate:** Milestone 2026.04.61 — Phase 2 JEPA retrain (Exp 798 → feeds Exp 799).

### ExecVerify: White-Box RL with Verifiable Stepwise Rewards for Code Execution Reasoning
- **Paper:** arXiv 2603.11226 (March 2026)
- **What:** Incorporates verifiable white-box rewards from code execution traces, rewarding both
  intermediate execution steps and final outputs. Enables RL training with ground-truth step-level
  feedback. Achieves SOTA on LiveCodeBench with verified step rewards vs outcome-only training.
- **Relevance to Carnot:** Code repair (SOTA GGUF code repair) relies on static energy verification.
  ExecVerify shows execution traces give provable ground-truth constraint labels: "variable undefined
  at step 3" is a hard constraint violation, not a probabilistic energy score. This maps to a new
  code-specific constraint type: ExecutionTraceConstraint. Complementary to SymCodeVerifier (Tier 2.5).
- **Concrete experiment:** Exp 801 — ExecVerifyCodeConstraintMiner: run test suite on LLM-generated
  code, capture execution trace (variable bindings, exception sites, failed assertions), convert
  each trace event to an IsingEBM constraint. Compare: CodeExtractor alone vs CodeExtractor + trace
  constraints on HumanEval. Target: trace constraints catch errors static analysis misses.
- **When to incorporate:** Milestone 2026.04.61 — Phase 4 code repair (Exp 801).

### PROGRS: Outcome-Guided Process Rewards with Coherence-Based Weighting
- **Paper:** arXiv 2604.02341 (April 2026)
- **What:** Introduces PROGRS combining process reward models with outcome verification via
  outcome-conditioned centering (rescale step rewards relative to final answer correctness)
  and coherence-based weighting (upweight steps with high inter-rater agreement). Resolves
  PRM-outcome misalignment. Outperforms standard PRMs by 2.8% on MATH-500.
- **Relevance to Carnot:** Exp 789 (EBM calibration) achieved ECE reduction 67.6% via isotonic
  regression but this is post-hoc. PROGRS's outcome-conditioned centering is a training-time
  calibration technique: during EORM training, weight each step's loss by whether the final
  response was correct. This makes EORM energy scores naturally correlate with correctness rates
  without post-hoc recalibration.
- **Concrete experiment:** Enhancement to EORM training — apply outcome-conditioned centering
  during JEPA v21 retrain (Exp 799): weight loss for step i by correct_final_answer_probability.
  Steps in correct responses get higher weight than steps in incorrect ones, even if the step
  itself is correct.
- **When to incorporate:** Milestone 2026.04.61 — integrated into JEPA v21 retrain (Exp 799).

### MARS: Margin-Aware Speculative Verification for Efficient LLM Verification
- **Paper:** arXiv 2601.15498 (January 2026)
- **What:** Proposes Margin-Aware Speculative Verification (MARS), a training-free fast-path
  verification strategy that uses target model logit margins as a confidence signal. When margin
  is high (model is decisive), accept without expensive verification. When margin is low (model
  is uncertain), run full verification. Achieves 2-3x speedup at acceptance rate 0.6-0.8.
- **Relevance to Carnot:** Carnot's cascade already does something similar (Tier 0 → Tier 3),
  but the decision to run expensive Ising verification is rule-based (tier thresholds). MARS
  margin signal — directly derived from the LLM's own logit distribution — could replace or
  augment the EORM energy threshold as a gating signal. High-margin LLM outputs rarely have
  constraint violations; low-margin outputs are where verification adds value.
- **Concrete experiment:** Enhance Exp 796 (SOTA GGUF code repair) — add MARS margin gate:
  extract logit margin from generated code tokens, skip expensive test execution for high-margin
  outputs. Measure oracle_calls_saved vs pass@1 degradation.
- **When to incorporate:** Milestone 2026.04.61 — augmentation to code repair pipeline.

## 2026-04-23 arxiv Scan (Milestone 2026.04.60 Planning)

### The Energy of Falsehood: Detecting Hallucinations via Diffusion Model Likelihoods
- **Paper:** arXiv 2602.11364 (February 2026)
- **What:** Introduces Semantic Energy as a metric measuring semantic divergence between original
  claims and reconstructions using discrete text diffusion. A Generative Stress Test corrupts
  claims with noise and reconstructs them; the reconstruction error (energy) detects hallucinations.
  Achieves AUROC 0.725 on FEVER, outperforming entropy-based baselines by 1.5pp with no labels.
- **Relevance to Carnot:** Directly validates energy as a detection signal beyond entropy.
  The "reconstruction energy" framing maps onto Carnot's IsingEBM: high energy = poor reconstruction
  = hallucination. Could unify Carnot's constraint energy with diffusion reconstruction energy as a
  single detection signal without requiring constraint extraction.
- **Concrete experiment:** Exp 789 — CalibrationAlignmentEBM: compare Carnot Ising energy scores
  against diffusion-style reconstruction energy on FoVer v2 steps. Measure which signal better
  separates correct from incorrect CoT steps. Combine both signals via learned linear fusion.
- **When to incorporate:** Milestone 2026.04.60 — Phase 3 calibration/energy research (Exp 789).

### EDU-PRM: Entropy-Driven Uncertainty for Process Reward Modeling
- **Paper:** arXiv 2503.22233 (March 2025)
- **What:** Entropy-driven framework for PRM training that uses only 1.5% of full training data.
  Uncertainty-aligned step segmentation automatically identifies which CoT steps carry the most
  labeling signal — steps near the decision boundary (high entropy across model samples).
  Outperforms public PRM baselines on ProcessBench despite the fraction of training data used.
- **Relevance to Carnot:** JEPA v19/v20 training uses FoVer-labeled CoT steps, but the labeling
  may produce imbalanced examples (high step_correct rate for simple arithmetic). EDU-PRM's
  uncertainty sampling would select the hardest-to-label steps for Z3 re-verification, improving
  corpus quality without increasing annotation cost. The entropy-based signal also maps to EORM
  energy: steps where EORM is uncertain (energy in the middle band) are the most valuable for
  JEPA training. This directly addresses the JEPA OOD generalization problem.
- **Concrete experiment:** Exp 782 — EDUPRMStepSelector: for each step in the FoVer v2 corpus,
  compute model prediction variance (bootstrap over training data). Select the top 30% highest-
  variance steps for JEPA v20 training. Compare OOD AUC: JEPA trained on EDU-selected steps
  vs uniform sampling. Target: OOD AUC improvement without more data labels.
- **When to incorporate:** Milestone 2026.04.60 — Phase 1 (Exp 782), feeds JEPA v20 retrain.

### Know When You're Wrong: Aligning Confidence with Correctness for LLM Error Detection
- **Paper:** arXiv 2603.06604 (March 2026)
- **What:** Framework for enabling LLMs to reliably signal uncertainty via output probabilities.
  Key finding: supervised fine-tuning (MLE) yields well-calibrated confidence; RL methods induce
  overconfidence. Calibration gap between expressed confidence and correctness is 15-25pp across
  tasks. Critical insight for designing reliable verifiers: probability ≠ correctness.
- **Relevance to Carnot:** Carnot's energy function is a calibration target: low energy = high
  confidence correct. If we align Carnot energy with actual correctness rates, we get better-
  calibrated verification. This is orthogonal to precision/recall — it's about the energy score
  being meaningful, not just discriminative. Addresses the model-size precision ceiling: larger
  models have higher confidence but our FP rate doesn't decrease, because we don't calibrate.
- **Concrete experiment:** Exp 789 — CalibrateEBMEnergy: for Qwen3.5-0.8B, compute Carnot
  energy scores on 200 GSM8K questions. Bucket by energy decile. Measure actual accuracy per
  bucket. If energy is well-calibrated, accuracy should increase monotonically with decreasing
  energy. Use isotonic regression to re-calibrate energy → probability. Report Expected
  Calibration Error (ECE) before and after calibration.
- **When to incorporate:** Milestone 2026.04.60 — Phase 3 (Exp 789).

### S*: Test Time Scaling for Code Generation via Energy-Ranked Candidate Selection
- **Paper:** arXiv 2502.14382 (February 2026)
- **What:** Hybrid test-time scaling combining parallel and sequential scaling with an adaptive
  selection mechanism. Uses distinguishing inputs (test cases that distinguish between candidates)
  for pairwise comparison. Enables a 3B model to outperform GPT-4o-mini on LiveCodeBench with
  sufficient compute budget. The selection mechanism is the key contribution: not just BoN, but
  structured tournament selection via execution.
- **Relevance to Carnot:** Exp 773 showed Carnot has 6x fewer oracle calls than SETS for the
  same pass rate. S* extends this further: instead of generating N candidates and taking the lowest
  energy, S* uses distinguishing inputs to separate them. Carnot energy can replace the execution-
  based selection mechanism for fast pre-filtering before running expensive tests. This could
  reduce distinguishing-input generation by 50%: use energy to prune candidates before tests.
- **Concrete experiment:** Exp 787 — SStarEnergyRanking: implement S*-style tournament selection
  for HumanEval. Replace distinguishing-input test with Carnot energy pre-filter: eliminate
  candidates with energy > threshold before running execution tests. Compare: pass@1, n_tests_run,
  wall_time_s vs pure S* (execution-only) and pure Carnot (energy-only).
- **When to incorporate:** Milestone 2026.04.60 — Phase 2 (Exp 787).

### Adaptive Test-Time Compute Allocation via Constrained Policy Optimization
- **Paper:** arXiv 2604.14853 (April 2026)
- **What:** Frames adaptive test-time compute allocation as constrained optimization. Reduces
  globally coupled budget-constrained programs to supervised classification via Lagrangian duality.
  Key finding: no single scaling strategy universally dominates — optimal strategy depends on
  per-instance difficulty. Achieves monotonic improvement with compute budget on MATH/GSM8K.
- **Relevance to Carnot:** Carnot's cascade already allocates compute adaptively (fast Tier 0
  gates to avoid expensive Tier 3 Ising). The constrained policy formulation provides a principled
  way to learn which gates to apply for which question types. Carnot energy is a natural difficulty
  signal: high energy questions need more verification compute; low energy can skip.
- **Concrete experiment:** Add to Exp 788 (Constraint Memory) — use constrained policy to learn
  which constraint types to add based on question energy profile. High-energy arithmetic questions
  → add carry-check constraint. High-energy semantic questions → add consistency constraint.
- **When to incorporate:** Milestone 2026.04.60 — as augmentation to Exp 788.

### Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning
- **Paper:** arXiv 2601.17223 (January 2026)
- **What:** Extends outcome verification to process-level verification, providing step-wise
  correctness signals with formal guarantees. Introduces verifiable PRM that maintains formal
  correctness proofs throughout inference. Key mechanism: each verification step either
  provably confirms or disproves partial solution correctness via SMT constraints.
- **Relevance to Carnot:** Carnot's SymCodeVerifier (Tier 2.5) and CausalReasoningVerifier
  (Tier 2.7) are partial implementations of this concept. This paper provides the formal
  framework: encode each CoT step as a logical formula, verify via SMT (Z3). Difference
  from current approach: step verification carries proof certificates, not just boolean flags.
  This enables the repair step to receive structured proof-based feedback ("step 3 unsat
  under constraint set {C1, C2}") rather than raw energy signals.
- **Concrete experiment:** Enhancement to Exp 788 — add proof certificate tracking to the
  constraint memory: when Z3 proves step_i UNSAT, store which constraints caused UNSAT.
  These become the "constraint memory" patterns for Tier 2 memory → constraint generation.
- **When to incorporate:** Milestone 2026.04.60 — integrated into Exp 788 design.

## 2026-04-23 arxiv Scan (Milestone 2026.04.58 Planning)

### Detecting and Correcting Hallucinations in LLM-Generated Code via Deterministic AST Analysis
- **Paper:** arXiv 2601.19106 (January 2026)
- **What:** Proposes a post-processing framework that parses generated code into an Abstract
  Syntax Tree (AST) and validates it against a dynamically-generated Knowledge Base built via
  library introspection. Deterministic rules find and fix API-level and identifier-level
  hallucinations (Knowledge Conflicting Hallucinations / KCHs). On 200 manually-curated
  Python snippets: 100% precision, 87.6% recall, 77.0% auto-correction rate.
- **Relevance to Carnot:** Carnot's CodeExtractor uses regex-based extraction which misses
  API-level hallucinations. AST-based verification is execution-free (unlike runtime
  instrumentation) and formally verifiable (unlike LLM re-checking). AST knowledge validation
  complements CodeExtractor: use AST for fast pre-filter (catches KCHs), CodeExtractor for
  arithmetic/logic violations. The 100% precision is the key property — zero FP rate means
  every detected violation is real, directly addressing the FP-rate problem at larger models.
- **Concrete experiment:** Exp 764 — ASTKnowledgeVerifier: implement AST parser + library
  introspection KB for Python code. Test on HumanEval subset: compare KCH detection rate vs
  CodeExtractor, measure precision/recall, integrate as Tier 0d pre-filter for code tasks.
- **When to incorporate:** Milestone 2026.04.58 — Phase 4 new research (Exp 764).

### Process Reward Models That Think
- **Paper:** arXiv 2504.16828 (April 2025, late-2025 ICLR workshop)
- **What:** Builds PRMs as verbalized step-wise reward models — instead of a discriminative
  head, the PRM generates a chain-of-thought verification for each step, explaining WHY the
  step is correct or incorrect. Requires orders of magnitude fewer process labels than
  discriminative PRMs. Achieves SOTA on ProcessBench with fewer than 1,000 labeled steps.
- **Relevance to Carnot:** EORM is a discriminative step-level oracle (binary correct/incorrect).
  A verbalized PRM would explain the error in natural language, enabling the repair step to
  receive structured feedback ("arithmetic carry is wrong in step 3 because 47+28=75, not 76").
  This directly addresses the repair quality problem: right now repair prompts include the
  raw violation signal; verbalized PRM would translate it to human-readable targeted feedback.
  Filed for EORM upgrade: add a verbalized explanation head alongside the energy score.
- **Concrete experiment:** Exp 765 Phase 4 enhancement — VerbalizePRM: train EORM with
  verbalized explanation head on FoVer v2 steps. Compare: does verbalized feedback improve
  repair accuracy vs raw energy signal? Measure: pass@1 improvement on HumanEval 2-round repair.
- **When to incorporate:** Milestone 2026.04.58 — enhancement of Exp 759/765.

### PPSEBM: Progressive Parameter Selection with Energy-Based Model for Continual Learning
- **Paper:** arXiv 2512.15658 (December 2025)
- **What:** Integrates an EBM with Progressive Parameter Selection (PPS) to prevent catastrophic
  forgetting in continual NLP tasks. The EBM generates replay data from previous tasks; this
  data guides PPS to focus parameter updates on task-relevant components. Best performance
  83.4 when combining EBM + PPS.
- **Relevance to Carnot:** PSV self-play has a recurring relapse problem (fp_rate_slope reverses
  from negative to positive across milestones). PPSEBM's approach — use energy to identify
  which constraint parameters are task-relevant and freeze others during adaptation — directly
  addresses the PSV coupling decay hypothesis. When adapting to new questions, freeze
  constraints that are already well-calibrated (low energy variance) and update only those
  with high variance. This prevents the self-play loop from overwriting learned couplings.
- **Concrete experiment:** Exp 762 — PPSConstraintSelector: EBM-guided progressive constraint
  parameter freezing during PSV self-play. Gate: fp_rate_slope < 0 after 30 self-play steps.
- **When to incorporate:** Milestone 2026.04.58 — Phase 4, PSV relapse fix via PPS (Exp 762).

### Stabilizing Iterative Self-Training with Verified Reasoning via Symbolic Recursive Self-Alignment
- **Paper:** arXiv 2603.21558 (March 2026)
- **What:** Proposes Symbolic Recursive Self-Alignment (SRSA) — uses a symbolic verifier (Z3,
  SMT solver) to screen self-generated training data before incorporating it into the next
  training round. Self-training instability (mode collapse, reward hacking) is mitigated by
  only accepting samples where the verifier confirms correctness. Achieves stable multi-round
  self-improvement on GSM8K and MATH.
- **Relevance to Carnot:** PSV self-play produces its own training data via VR repairs; the
  relapse problem (fp_rate reversal) is likely caused by incorporating incorrect self-generated
  repairs into the constraint memory. SRSA's symbolic screening approach maps directly:
  before writing a VR repair to the session memory, verify it with Z3/SymCodeVerifier first.
  Only repairs that pass Z3 verification enter the memory pool; others are discarded.
  This is the missing gate in the FR-11 relay: currently all VR outputs write to memory,
  including false positives and failed repairs.
- **Concrete experiment:** Exp 756 — SRSAMemoryGate: add Z3 verification gate before
  session_memory.write() in the FR-11 relay pipeline. Only Z3-verified repairs enter memory.
  Test: does PSV fp_rate_slope return to negative with memory gating?
- **When to incorporate:** Milestone 2026.04.58 — PSV relapse fix (Exp 756).

### Spark: Stepwise Process-Aware Rewards for Reference-Free Process Reinforcement Learning
- **Paper:** arXiv 2512.03244 (December 2025)
- **What:** Three-stage framework: generator produces solutions; verifier evaluates each step
  using parallel scaling (self-consistency) and sequential scaling (meta-critique); verification
  outputs become synthetic PRM training data. Achieves 67.5 F1 on ProcessBench vs 66.4 for
  reference-guided training, with NO ground-truth process labels needed.
- **Relevance to Carnot:** PSV self-play is similar but uses energy scores instead of LLM
  meta-critique. Incorporating the Spark meta-critique layer as a secondary verifier signal
  could stabilize PSV training by weighting high-consensus (self-consistent) repairs more
  strongly in the memory pool. Filed as complementary to SRSA memory gating.
- **Concrete experiment:** Enhancement to Exp 756 — add self-consistency weighting for PSV
  memory writes: run each VR repair K=3 times (cheap with CPU inference), only write to
  memory if K≥2 agree. Stabilizes memory without needing Z3 for every repair.
- **When to incorporate:** Milestone 2026.04.58 — Exp 756 PSV recovery enhancement.

### Recurrent-Depth Transformers (RDT) — Phase 3 EBT Architecture Primitive
- **Repo:** https://github.com/kyegomez/OpenMythos (speculative reconstruction of Claude Mythos;
  9.3k stars but 35 commits — treat code as unvetted; read the underlying papers it cites)
- **What:** A transformer architecture where a small set of layers is applied iteratively in a
  recurrent block between a prelude and coda. The recurrent update
  `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` reinjects the encoded input at every step and
  keeps the injection matrix LTI-constrained (spectral radius < 1) to guarantee iterative
  convergence. Adaptive Computation Time (ACT) halts iteration when refinement plateaus.
  Loop-index positional embeddings let the network distinguish early vs late refinement steps.
- **Relevance to Carnot:** Phase 3 (EBM/EBT foundation model) targets continuous-latent,
  non-autoregressive, self-correcting reasoning. RDT is architecturally isomorphic: the
  recurrent update can be viewed as implicit gradient descent on an energy function, and the
  LTI stability constraint is exactly the kind of convergence proof an EBT needs. Specific
  transferable primitives: (1) LTI spectral-radius constraint on any iterative refinement
  block; (2) loop-index PE to parameterize "refinement depth"; (3) ACT-style halting as a
  learned proxy for "energy converged." Not for Phase 1 (verify-repair) — it's a Phase 3
  design reference.
- **Concrete experiment:** Not now. When Phase 3 EBT architecture work begins (post–2026.Q3),
  first-pass design should instantiate a 3-stage (prelude/recurrent/coda) EBT, bind the
  recurrent iteration count to an energy-convergence criterion rather than ACT, and use the
  LTI constraint on the injection parameters. Benchmark against a vanilla stacked-transformer
  EBT for parameter efficiency and reasoning-task scaling.
- **Credibility caveat:** The OpenMythos code itself is unvetted (prolific-author pattern;
  hype-to-commit ratio concerning; no Claude-comparison benchmarks). The ideas are real and
  cited; the implementation should not be treated as reference. Track the underlying
  recurrent-depth-transformer papers (Geiping et al. on looped transformers, Banino et al. on
  PonderNet/ACT) directly.
- **When to incorporate:** Phase 3 kickoff (no specific milestone yet; Phase 1 + 2 still
  active). Filed as an architectural primitive to consider when foundation-model work begins.

## 2026-04-22 arxiv Scan (Milestone 2026.04.56 Planning)

### Efficient Test-Time Scaling via Probing Internal States of LLMs
- **Paper:** arXiv 2511.06209 (Nov 2025, revised Jan 2026)
- **What:** Trains a small transformer probe (<10M params) on LLM hidden states to score
  reasoning step credibility, achieving parity with much larger PRMs on math, planning, and QA
  without domain-specific annotation. Multi-step probe outperforms single-step designs.
- **Relevance to Carnot:** JEPAReasonerProbe (Exp 726, AUC=1.0, Tier 2.1) is a single-step
  probe. This paper shows multi-step probes generalise better to unseen domains and harder
  problems. Upgrading to multi-step probe architecture (pool features across multiple CoT
  steps) should improve OOD robustness without requiring more labelled data.
- **Concrete experiment:** Exp 738 — train step-level latent probe on FoVer v2 using pooled
  hidden states across all CoT steps; compare step-F1 and OOD AUC against query-level
  JEPAReasonerProbe.
- **When to incorporate:** Milestone 2026.04.56 — Phase 6 new research (Exp 738).

### LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
- **Paper:** arXiv 2602.09924 (Feb 10, 2026). ICLR 2026 Workshop on Latent and Implicit Thinking.
- **What:** Linear probes trained on pre-generation activations (before any output token) predict
  task success on math/coding at substantially better-than-chance accuracy. Used to route
  queries across models, reducing inference cost 70% on MATH.
- **Relevance to Carnot:** Provides theoretical grounding for Tier 2.1 (JEPAReasonerProbe) and
  motivates using pre-generation probe predictions to gate whether a query even needs CoT
  generation — a deeper efficiency win than skipping later tiers. Confirms JEPAReasonerProbe's
  mechanism.
- **When to incorporate:** Milestone 2026.04.56 — motivation for Exp 732 cross-validation;
  cite as theory for why Tier 2.1 works.

### Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations
- **Paper:** arXiv 2601.07422 (Jan 12, 2026). Accepted to ACL 2026.
- **What:** Identifies two distinct internal representation pathways for hallucination signals
  in LLMs — one anchored to questions, one to answers — using token patching. The model's
  internal states are aware of truthfulness even when it hallucinates.
- **Relevance to Carnot:** Current probes (SinkProbe, HalluField) use single-pathway architectures.
  Dual-pathway fusion probe could improve AUROC by capturing both question-anchored and
  answer-anchored hallucination signals. Filed for .57 probe architecture upgrade.
- **When to incorporate:** Milestone 2026.04.57 — probe architecture upgrade.

### Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling
- **Paper:** arXiv 2604.13386 (April 15, 2026).
- **What:** Single-layer probes for detecting model deception are unreliable; multi-layer
  ensembles improve AUROC by +29-78%. Accuracy scales ~5% AUROC per 10x parameter increase.
- **Relevance to Carnot:** All Carnot latent probes are currently single-layer. Multi-layer
  ensemble with residual connections would improve reliability, especially on OOD benchmarks.
  Filed for .57 probe upgrade.
- **When to incorporate:** Milestone 2026.04.57 — probe architecture upgrade.

### Process Reward Models Meet Planning: PDDL-Derived Step-Level Labels at Scale
- **Paper:** arXiv 2604.17957 (April 2026). Accepted to ACL 2026.
- **What:** Uses PDDL planning problems as a scalable synthetic source for PRM training data,
  generating ~1M labeled reasoning steps. FoVer v2 already uses PDDL for labeling.
- **Relevance to Carnot:** Validates the FoVer PDDL labeling approach and provides a blueprint
  for scaling FoVer v2 from ~1K to ~100K labeled steps. Larger labeled corpus would improve
  JEPAReasonerProbe, EORM, and PrivacyFilter KAN training.
- **When to incorporate:** Milestone 2026.04.57 — FoVer v3 corpus scaling.

### IPVRM: Prefix-Value Learning for Step-Level Process Reward Models
- **Paper:** arXiv 2604.13197 (April 14, 2026).
- **What:** Prefix-conditioned value function estimating step correctness via TD-difference
  learning. Fixes train-inference mismatch in step-level scoring. Consistent improvements
  on ProcessBench vs. cross-entropy classification heads.
- **Relevance to Carnot:** EORM currently uses a direct classification head. IPVRM's TD
  formulation maps to energy function: E(step) = -log P(correct | prefix). Filed for .57
  EORM architecture upgrade.
- **When to incorporate:** Milestone 2026.04.57 — EORM TD retraining.

### Kaiwu-PyTorch-Plugin: PyTorch Plugin for Coherent Ising Machine Acceleration
- **Paper:** arXiv 2602.19114 (Feb 22, 2026).
- **What:** Integrates a Coherent Ising Machine (CIM) into the PyTorch EBM training loop.
  Accelerates Boltzmann sampling, active data selection. Tests on biological and text datasets.
- **Relevance to Carnot:** After KV260 FPGA experiments conclude, CIM cloud access is the
  next hardware acceleration tier. KPP's PyTorch API would map onto `carnot-ising` via Python
  FFI, consistent with the existing SamplerBackend abstraction.
- **When to incorporate:** Milestone 2026.04.58+ — post-FPGA hardware path.

### Unified Performance-Cost Landscape of Parallel p-bit Ising Machines
- **Paper:** arXiv 2604.01564 (April 2, 2026).
- **What:** Synchronous p-bit designs with 3-4 bit DAC resolution achieve comparable MaxCut
  quality at less than half the hardware cost vs. asynchronous. Evaluated on G-set benchmarks.
- **Relevance to Carnot:** Directly informs the KV260 FPGA register-level design. Synchronous
  + 4-bit fixed-point p-bits are the optimal hardware target for Carnot's Ising tier.
  Ising Sampler v3 RTL (hardware/kv260/ising_sampler_v3.v from Exp 662) should adopt
  synchronous 4-bit p-bits per this paper's recommendation.
- **When to incorporate:** Next FPGA synthesis milestone (when Vivado/yosys installed).

### FunPRM: Function-as-Step PRM with Meta Reward Correction for Code Generation
- **Paper:** arXiv 2601.22249 (January 2026).
- **What:** Treats code functions as CoT steps, uses unit-test outcomes to denoise noisy step
  rewards via meta-learning. SOTA on LiveCodeBench.
- **Relevance to Carnot:** KAN Tier-1 is trained on step-level correctness labels. Meta-reward
  correction could denoise weakly-labeled FoVer steps where PDDL ground truth is unavailable.
- **When to incorporate:** Milestone 2026.04.57 — KAN v4 training improvement.

## 2026-04-22 arxiv Scan (Milestone 2026.04.55 Planning)

### JEPA-Reasoner — Latent-Space Reasoning Verification (Pre-generative)
- **Paper:** arXiv 2512.19171 (December 2025)
- **What:** JEPA-Reasoner decouples latent-space reasoning from linguistic reconstruction. A
  JEPA-style predictor operates entirely in hidden-state space, predicting whether a reasoning
  step is logically consistent with prior context without generating any tokens. Achieves
  comparable accuracy to full rollout verification at <1ms per step.
- **Relevance to Carnot:** Carnot's JEPA predictor (Tier 2) uses partial response text as
  input. JEPA-Reasoner shows that operating directly in latent space — skipping text
  re-tokenization — is both faster and more accurate. Applied to Carnot: instead of
  encoding step_text → embedding → MLP → score, compute score directly from the LLM's
  hidden state at the end of each reasoning step. This is the natural Tier 0h: a
  pre-generative latent probe trained on FoVer v2 pairs.
- **Concrete experiment:** Exp 726 — JEPAReasonerProbe: extract hidden states at each CoT
  step boundary from Qwen3.5-0.8B, train 2-layer MLP on FoVer v2 pairs, compare AUC
  vs EORM (text-based Tier 2). If AUC >= 0.75, propose as Tier 0h.
- **When to incorporate:** Milestone 2026.04.55 — Phase 4 new research (Exp 726).

### Variable Granularity Verification — Adaptive Compute for Test-Time Scaling
- **Paper:** arXiv 2505.11730 (May 2025)
- **What:** Variable Granularity Search (VGS) dynamically selects verification granularity
  per reasoning instance: skip expensive verification for high-confidence responses, run
  deep multi-step checking for uncertain ones. Reduces average verification cost 3-5x with
  <1% accuracy degradation on MATH and GSM8K benchmarks.
- **Relevance to Carnot:** Carnot's cascade currently runs ALL tiers for every response.
  VGS maps directly: use EORM energy as the confidence gate. If EORM energy < threshold:
  skip Ising and return. Only run full Ising verification for responses where EORM is
  uncertain (energy in [low_threshold, high_threshold]). This is orthogonal to the existing
  early-exit architecture (Tier 0 → Tier 1 → Tier 2 → Tier 3) and could reduce Tier 3
  Ising calls by 60-80% on high-confidence responses.
- **Concrete experiment:** Exp 727 — VariableGranularityGate: add EORM confidence interval
  gate before Tier 3 Ising. Set thresholds from FoVer v2 calibration. Measure: what
  fraction of correct responses skip Ising verification? Target: >50% skip rate with
  <5% FN degradation.
- **When to incorporate:** Milestone 2026.04.55 — Phase 4 new research (Exp 727).

### ActPRM — Active Learning for Efficient Process Reward Model Annotation
- **Paper:** arXiv 2504.10559 (April 2026)
- **What:** ActPRM uses entropy-driven uncertainty sampling to select the most informative
  CoT steps for annotation. Achieves comparable PRM quality to full annotation using only
  50% of steps. Key insight: steps with high uncertainty contribute most to calibration;
  steps near the decision boundary are most valuable for training signal.
- **Relevance to Carnot:** FoVer v2 provides 1000+ pairs but the PDDL labeling may produce
  imbalanced examples (high step_correct rate for simple arithmetic). ActPRM's uncertainty
  sampling would select the hardest-to-label steps for Z3 re-verification, improving
  corpus quality without increasing annotation cost. Applies to JEPA v18 training: use
  ActPRM to weight the LambdaRank training loss toward uncertain examples.
- **Concrete experiment:** Wire into JEPA v18 training (Exp 717): weight each training pair
  by its uncertainty score (model confidence variance across FoVer v2 bootstrap samples).
  Compare OOD AUC: JEPA v18 with vs without ActPRM weighting.
- **When to incorporate:** Milestone 2026.04.55 — JEPA v18 training (Exp 717, as optional weighting).

### Constraint-Induced Distortion in Reasoning LLMs
- **Paper:** arXiv 2601.01490 (January 2026)
- **What:** Shows that LLMs under strict constraint compliance sacrifice factual accuracy
  to satisfy constraints — a "distortion" effect where the model satisfies formal requirements
  at the cost of semantic correctness. Models with stronger priors (more capable/larger)
  show MORE distortion because they try harder to satisfy constraints. Demonstrates that
  pure constraint enforcement without calibration degrades accuracy on capable models.
- **Relevance to Carnot:** Explains why VR degrades Gemma4-E4B-it (more capable model) but
  helps Qwen3.5-0.8B (weaker model). The adaptive threshold gating (Exp 707) is the right
  mitigation but the constraint semantics also need adjustment. For capable models: use
  weaker constraints (advisory signals rather than hard gates), prefer abstention over
  repair when uncertainty is high. Filed as root-cause explanation for Gemma4 degradation.
- **Concrete experiment:** Exp 721 — Gemma4 graduated threshold search: test thresholds
  [0.10, 0.20, 0.30, 0.40] and abstain (skip repair) when violation confidence < threshold.
  Target: find the threshold where signed_improvement > 0 for Gemma4.
- **When to incorporate:** Milestone 2026.04.55 — Gemma4 threshold tuning (Exp 721).

## 2026-04-22 arxiv Scan (Milestone 2026.04.53 Planning)

### KAN Formal Verification — MILP-Based Provable Energy Monotonicity
- **Paper:** arXiv 2602.06737 (February 2026)
- **What:** Presents formal verification for Kolmogorov-Arnold Networks via Mixed Integer
  Linear Programming (MILP) encodings of piecewise-affine KAN abstractions. Enables sound
  property checking (e.g., energy monotonicity) with provable correctness guarantees.
- **Relevance to Carnot:** Carnot's KAN energy tier currently verifies via sampling (MCMC)
  rather than formal proof. MILP-based verification could provide sound energy bounds:
  "E(wrong) > E(correct) holds for all inputs in this constraint region," providing
  a formal certificate for the KAN energy tier's discriminative power.
- **Concrete experiment:** Apply MILP verification to carnot-kan energy outputs on 50
  GSM8K correct/incorrect pairs. Verify energy monotonicity property. Measure: does formal
  MILP verification catch the same violations as Ising sampling, faster?
- **When to incorporate:** Milestone 2026.04.53 — Phase 4 new research (Exp 700).

### PaCoRe — Parallel Coordinated Reasoning for Test-Time Compute Scaling
- **Paper:** arXiv 2601.05593 (January 2026)
- **What:** Parallel coordinated reasoning where N independent LLM chains sample diverse
  solutions, then learned coordination functions merge via energy-weighted voting. Reduces
  latency vs. sequential chain-of-thought while maintaining quality at larger N.
- **Relevance to Carnot:** The PSV self-play loop runs one chain at a time. PaCoRe-style
  parallel chains could scale PSV to K=4 parallel VR attempts per question, using energy
  scores to select the best-verified response. With DualGPU confirmed (Exp 685), two chains
  per GPU is feasible. Complements Tier 3 self-learning: diverse solutions give better
  coverage of constraint violation types.
- **Concrete experiment:** Adapt PSV iteration to run K=2 parallel chains per GPU; merge
  via SymCodeVerifier energy vote (lowest violation energy wins). Compare: does K=2 parallel
  improve signed_improvement vs. single-chain PSV iteration?
- **When to incorporate:** Milestone 2026.04.53 — incorporated into PSV self-play (Exp 697).

### Formal Step Intermediaries for Verifiable Reasoning
- **Paper:** arXiv 2603.29500 (March 2026)
- **What:** Trains LLMs to generate formal logic intermediaries (FOL propositions) alongside
  each CoT step, enabling Z3 verification of step entailment. Achieves near-zero false
  positive rates by requiring steps to be formally entailed by prior premises.
- **Relevance to Carnot:** Extends SymCodeVerifier (Tier 2.5) and CausalReasoningVerifier
  (Tier 2.7). Instead of verifying final arithmetic (SymCode) or carry-forward (Causal),
  this approach verifies each intermediate step via Z3 entailment checking. Could be the
  new Tier 2.8 candidate alongside Eidoku (arXiv 2512.20664).
- **Concrete experiment:** Implement Tier 2.8: FormalStepVerifier that asks the model to
  output one FOL proposition per step, then verifies entailment with Z3. Test on FOVER
  corpus — compare AUC to SymCodeVerifier and CausalReasoningVerifier.
- **When to incorporate:** Milestone 2026.04.53 — Tier 2.8 candidate (Exp 695).

### Hard Constraints + Soft Generation Hybrid Decoding
- **Paper:** arXiv 2602.01090 (February 2026)
- **What:** Combines hard constraint enforcement (grammar masks, feasibility filters) with
  soft generative decoding. Guarantees feasible outputs for combinatorial optimization
  problems while maintaining generative diversity.
- **Relevance to Carnot:** Carnot's structured-equation forcing (Exp 653) is purely
  prompt-based (soft). This paper provides the architecture for HARD constraint enforcement
  during decoding — token-level masking of outputs that would violate arithmetic structure.
  Applied to the COMPUTE: format: when the model starts generating "COMPUTE: A op B =",
  constrain the next tokens to valid arithmetic expressions.
- **Concrete experiment:** Implement token-level grammar masking for COMPUTE: lines. Compare:
  do hard-constrained COMPUTE: expressions reduce SymCodeVerifier violation rate vs. soft
  prompt-only forcing? Measure on 50 GSM8K questions.
- **When to incorporate:** Milestone 2026.04.53 — refine structured-equation forcing in
  VR credibility hardening (filed for VR hard questions experiment, Exp 694).

### JEPA Latent Probing for Discrete Symbol Extraction
- **Paper:** arXiv 2603.20327 (March 2026)
- **What:** Probes V-JEPA latents for emergent discrete symbols (AI Mother Tongue),
  revealing spatiotemporal structure without supervision. Converts continuous JEPA latents
  to discrete symbol sequences, improving OOD generalization.
- **Relevance to Carnot:** Carnot's JEPA v15 exhibits OOD regression (AUC=0.4751 below
  random). The anti-correlation may stem from overly-continuous latent representations that
  don't generalize to unseen arithmetic patterns. Applying discrete symbol probing to
  JEPAPredictor's hidden layer could extract more compositional features, improving OOD AUC.
- **Concrete experiment:** Apply linear probing on JEPA latent layer to extract N=16 discrete
  symbols. Compare OOD AUC on GSM8K 500-699: discrete-symbol JEPA v16 vs. raw-latent v15.
- **When to incorporate:** Milestone 2026.04.53 — filed for JEPA v16 architecture (Exp 693).

## 2026-04-22 arxiv Scan (Milestone 2026.04.52 Planning)

### FoVer — Formal Verification for Scalable PRM Training Data Generation
- **Paper:** arXiv 2505.15960 (May 2025)
- **What:** Uses Z3 SMT solver and Isabelle to automatically annotate step-level error
  labels in PRM training data. Replaces costly human annotation with formal proofs —
  a step is labeled correct if and only if it is provably reachable from premises via
  valid arithmetic reasoning chains.
- **Relevance to Carnot:** Carnot's JEPA predictor (Tier 2) is trained on the
  hand-annotated FOVER corpus (57 live pairs as of .51), limiting training data
  diversity. FoVer provides a scalable, provably-correct annotation pipeline: run
  Qwen3.5-0.8B on 200 GSM8K questions, extract CoT steps, run Z3 on each step to
  determine correct/incorrect, generate fover_labeled_formal_v1.json. This could
  expand the training corpus 10x without human annotation, addressing the persistent
  training-data bottleneck for JEPA.
- **Concrete experiment:** Exp 686 (FoVer Formal PRM Labels): integrate Z3 step-level
  annotation pipeline. Run 200 GSM8K CoT chains, auto-label each step. Produce
  fover_labeled_formal_v1.json. Verify label agreement rate with existing FOVER
  hand-labels (should be > 80%). Use as JEPA v16 training data in next milestone.
- **When to incorporate:** Milestone 2026.04.52 — Phase 4 new research.

### HalluSAE — Sparse Auto-Encoder Feature Attribution for Hallucination Detection
- **Paper:** arXiv 2604.16430 (April 2026)
- **What:** Detects hallucinations via sparse auto-encoder decomposition of LLM hidden
  states. Identifies monosemantic features causally linked to factual errors. Achieves
  0.91+ AUROC on factual verification tasks by attributing energy spikes to specific
  learned features, not just aggregate hidden-state statistics.
- **Relevance to Carnot:** Carnot's EORM (Tier 2) scores full CoT sequences; it cannot
  attribute which specific features caused the energy spike. HalluSAE's sparse feature
  attribution would enable causal diagnosis: "this response flagged because Feature 47
  (off-by-one arithmetic) activated." This maps to Tier 2.5 feature attribution and
  could replace the NUP Probe v4 (Tier 0c) with a more interpretable, causally-grounded
  signal. Also relevant to explaining why structured-equation forcing reduces hallucinations.
- **Concrete experiment:** Exp 687 (HalluSAE Sparse AE): train a 512-dim sparse AE on
  Qwen3.5-0.8B hidden states from FOVER corpus. Identify top-10 hallucination-predictive
  features. Compare: do these features correlate with COMPUTE: line presence? If yes,
  provides mechanistic explanation for the VR win (Exp 668).
- **When to incorporate:** Milestone 2026.04.52 — Phase 4 new research (Exp 687).

### PSV — Self-Play via Formal Verification for Autonomous Learning
- **Paper:** arXiv 2512.18160 (December 2025)
- **What:** Propose, Solve, Verify (PSV) framework where a generator model proposes
  problems, a solver model attempts them, and formal verification provides binary
  correctness labels. Uses this labeled data for self-play to improve both problem
  quality and solution accuracy without human supervision.
- **Relevance to Carnot:** Directly implements Tier 3 of Carnot's self-learning roadmap
  (JEPA predictive verification). The PSV loop adapted to Carnot: (1) select GSM8K
  question variants, (2) run VR pipeline to verify/repair, (3) use verification verdict
  as a training signal for constraint weight updates and JEPA retraining. Unlike prior
  FR-11 experiments that relied on pre-collected violation pairs, PSV continuously
  generates fresh labeled data. The structured-equation forcing (Exp 653) makes the
  "solve" step verification-friendly.
- **Concrete experiment:** Exp 688 (PSV Self-Play Loop): implement a 10-iteration
  self-play loop where each iteration generates 20 questions, runs VR pipeline with
  structured forcing, uses binary correct/incorrect labels to update ConstraintTemplateLibrary
  weights (Tier 1 online learning). Measure: does constraint weight quality improve
  across iterations (reduction in FP rate)?
- **When to incorporate:** Milestone 2026.04.52 — Phase 4 new research (Exp 688).

### Eidoku — CSP-Based Structural Verification Gate for LLM Reasoning
- **Paper:** arXiv 2512.20664 (December 2025)
- **What:** Reformulates LLM output verification as a Constraint Satisfaction Problem
  operating independently of generation likelihood. Models feasibility checks via
  structural violation cost functions that are independent of model confidence — avoiding
  the calibration gap between high-confidence hallucinations and correctly-uncertain
  correct responses.
- **Relevance to Carnot:** Carnot's SymCodeVerifier (Tier 2.5) executes Python to check
  arithmetic — which is structure-blind (can't check logical consistency). Eidoku's CSP
  gate could complement SymCode: first check arithmetic correctness (SymCode), then check
  structural consistency via CSP (Eidoku). Filed for .53.
- **When to incorporate:** Milestone 2026.04.53 — Tier 2.8 candidate.

### I-CALM — Confidence-Aware Abstention for Hallucination Mitigation
- **Paper:** arXiv 2604.03904 (April 2026)
- **What:** Prompt-only intervention that reduces hallucination via abstention incentives.
  Teaches models to say "I don't know" when confidence is low via targeted system prompts.
  Dramatically reduces false positive hallucination repairs (model defers rather than
  guessing) without changing model weights.
- **Relevance to Carnot:** Carnot's repair pipeline currently always attempts repair when
  a violation is detected — even on low-confidence detections, which causes false positives.
  I-CALM's abstention pattern applied to repair: when energy is high but below a confidence
  threshold, output "abstain" rather than a potentially-wrong repair. Aligns with the false
  positive problem identified after the .51 VR win (post_accuracy=1.0 suspicious — may be
  abstentions counted as correct). Filed for .53.
- **When to incorporate:** Milestone 2026.04.53 — Repair abstention layer.

## 2026-04-21 arxiv Scan (Milestone 2026.04.51 Planning)

### IAS — Instance-Adaptive Scaling for Process Reward Model Uncertainty Calibration
- **Paper:** arXiv 2506.09338 (June 2025, NeurIPS 2025)
- **What:** Develops quantile regression calibration for PRMs to align confidence estimates with true success probabilities. Introduces Instance-Adaptive Scaling (IAS) that dynamically adjusts verification compute budget per reasoning step based on calibrated confidence bounds — high-confidence steps skip expensive verification, uncertain steps get deeper checking.
- **Relevance to Carnot:** Directly addresses Exp 655 gate failure: the fixed threshold=0.30 rejected VR #18 because symcode and hermes recall dragged the ensemble. IAS-style quantile calibration would set thresholds adaptively based on per-model error distributions rather than fixed global values. Carnot's EBM energy can replace PRM confidence: train a quantile regression head on FOVER pairs to predict 10th/90th percentile recall bounds, set gate threshold at the 10th percentile for a given model+domain. Enables dynamic gate calibration without retraining the gate for each new model.
- **Concrete experiment:** Exp 674 (IAS Adaptive Gate Calibration): train quantile regression head on FOVER pairs predicting verification recall bounds. Use calibrated 10th-percentile recall as the adaptive gate threshold. Compare: does the calibrated gate open on the .50 test set where fixed threshold=0.30 failed? If yes, this is a structural fix for repeated gate failures.
- **When to incorporate:** Milestone 2026.04.51 — Phase 5 new research (Exp 674).

### LOS-Net — Sequence-Level Hallucination Detection from Full Output Distributions
- **Paper:** arXiv 2503.14043 (March 2026)
- **What:** LOS-Net is a lightweight attention-based architecture trained on the full LLM Output Signature — the complete sequence of next-token distributions (not just argmax tokens). Achieves sub-100ms detection latency and strong transfer across datasets and model families. Demonstrates that the full distribution trajectory carries far more hallucination signal than any individual token's entropy.
- **Relevance to Carnot:** Carnot's Tier 0b (SpilledEnergyDetector) uses per-token logit discrepancy — a single-token signal. LOS-Net shows that the SEQUENCE of distributions contains an orthogonal, stronger signal. A sequence-level model over the full distribution trajectory could improve on SpilledEnergy's Tier 0b AUC and catch hallucinations that are invisible at the token level. The attention architecture is compact (< 5M params), FPGA-friendly, and the input (softmax distributions) is already computed during generation.
- **Concrete experiment:** Exp 675 (LOS-Net Sequence Detector): implement lightweight attention model over the sequence of top-K softmax distributions from Qwen3.5-0.8B generation. Train on 100 FOVER live pairs (from results/fover_labeled_steps_live.json). Compare AUC vs SpilledEnergyDetector (Tier 0b) and NUP Probe v4 (Tier 0c). If AUC >= 0.75, propose as Tier 0h candidate replacing Tier 0b.
- **When to incorporate:** Milestone 2026.04.51 — Phase 5 new research (Exp 675).

## 2026-04-21 arxiv Scan (Milestone 2026.04.50 Planning)

### SpecGuard — Verification-Aware Speculative Decoding for Step-Level Reasoning
- **Paper:** arXiv 2604.15244 (April 2026)
- **What:** SpecGuard performs step-level verification using only model-internal signals: log-probability-based verification (LPBV) scores the probability that the generated step is consistent with prior steps, and attention-based grounding verification (ABGV) checks whether tokens are grounded in the input. No external verifier calls required during generation — pure internal signal, sub-millisecond per step.
- **Relevance to Carnot:** Carnot's cascade currently needs SymCodeVerifier (Python eval) or HERMES v2 (LLM call) for step-level arithmetic verification — both require external compute. SpecGuard offers a Tier 0f signal that's faster than anything in the cascade: one dot product per step over frozen model weights. If ABGV attention signal correlates with arithmetic violations, it becomes the cheapest reliable early-exit before SinkProbe (Tier 1). The log-prob ratio (LPBV) is architecturally identical to SpilledEnergyDetector (Tier 0b) but applied at step boundaries rather than per-token.
- **Concrete experiment:** Exp 658 (SpecGuardVerifier): implement LPBV + ABGV step-boundary verification using cached logits from Qwen3.5-0.8B generation. Compute step-boundary log-prob ratio and attention concentration for each CoT step. Train a lightweight classifier on FOVER corpus. Compare AUC vs SymCodeVerifier on live_pairs_578.json. If AUC >= 0.70, deploy as Tier 0f.
- **When to incorporate:** Milestone 2026.04.50 — Phase 3 new research (Exp 658).

### HALP — Pre-Generative Hallucination Detection from Internal Model States
- **Paper:** arXiv 2603.05465 (March 2026)
- **What:** HALP predicts hallucination risk from pre-generative internal states (pooled visual features and decoder activations) using a simple MLP probe. Key insight: the model's internal state at the end of reading the question contains sufficient information to predict whether the response will hallucinate — before a single output token is generated. Achieves 0.93 AUROC without any token generation.
- **Relevance to Carnot:** Carnot's JEPA predictor (Tier 2) operates on partial responses (prefix fraction = 50%). HALP's pre-generative approach would move detection even earlier: predict at the question-end hidden state whether the model will produce an incorrect answer. This maps directly to Carnot's Tier 0 (before any generation): extract the question-end hidden state from Qwen3.5-0.8B, pass through a 2-layer MLP trained on FOVER corpus, get a pre-generation risk score. If score > threshold, prompt with a structured equation forcing hint before generation even starts.
- **Concrete experiment:** Exp 663 (HALPProbe): extract question-end decoder activations from Qwen3.5-0.8B on FOVER corpus. Train 2-layer MLP probe. Compare AUC vs EORM on same test set. If AUC >= 0.75, wire into VerifyRepairPipeline as Tier 0g (pre-generative gate).
- **When to incorporate:** Milestone 2026.04.50 — Phase 6 new research (Exp 663).

### LSEBMCL — Latent Space EBM for Continual Learning Without Catastrophic Forgetting
- **Paper:** arXiv 2501.05495 (January 2026)
- **What:** Uses an EBM to sample from the distribution of previous task data, enabling exact replay without storing raw examples. The EBM models the latent distribution of past examples; when training on new tasks, the EBM generates replay samples that prevent catastrophic forgetting. Demonstrated on NLP tasks including reasoning.
- **Relevance to Carnot:** Carnot's Tier 2 constraint memory (ConstraintTemplateLibrary) accumulates patterns across sessions but loses old patterns when new sessions overwrite weights. LSEBMCL's EBM replay applies directly: train an Ising EBM on the current session's violation distribution; when the next session starts, sample from the EBM to replay past violations alongside new ones. This maintains constraint coverage without storing raw examples — the energy function IS the memory. Directly implements Tier 2 of the self-learning roadmap.
- **Concrete experiment:** Exp 660 (LSEBMCLConstraintMemory): implement EBM replay for ConstraintTemplateLibrary. After session 1, train a small IsingEBM on the violation feature vectors. In session 2, sample from the EBM to augment new violations before updating template weights. Measure: does catastrophic forgetting reduce? Compare with/without EBM replay on 3 sequential sessions.
- **When to incorporate:** Milestone 2026.04.50 — Phase 4 self-learning (Exp 660).

## 2026-04-21 arxiv Scan (Milestone 2026.04.49 Planning)

### Parallel Densely Connected Probabilistic Ising Machine with Inertia
- **Paper:** arXiv 2604.17109 (April 2026)
- **What:** Introduces densely-connected probabilistic Ising machine dynamics with an inertia term: each spin's local field is exponentially smoothed h_i(t+1) = alpha*h_i(t) + (1-alpha)*sum_j(J_ij*s_j(t)). The inertia parameter alpha controls the momentum that prevents oscillation and accelerates convergence. Fully parallel update schedule (vs checkerboard alternating). Validated on real FPGA hardware with 35x speedup over sequential Gibbs.
- **Relevance to Carnot:** Carnot's KV260 FPGA work uses a checkerboard parallel Gibbs sampler (ising_sampler_v2.v). Inertia dynamics could be the v3 RTL: same area as v2, faster convergence for dense constraint graphs (which are common in multi-step arithmetic verification where many variable pairs interact). The Python simulation can be validated before RTL implementation. 35x hardware speedup > the 26x from D-Wave (Exp 598), making this the highest-speedup FPGA path identified so far.
- **Concrete experiment:** Exp 648 (ParallelDenseIsingInertia): implement Python simulation of inertia Ising dynamics (ParallelDenseIsingSampler with inertia alpha parameter). Benchmark convergence steps vs standard checkerboard Gibbs on 100-spin, 200-spin, 500-spin constraint graphs. Generate v3 RTL specification for future FPGA synthesis. Target: convergence steps reduced by >= 20% vs v2 checkerboard.
- **When to incorporate:** Milestone 2026.04.49 — Phase 4 new research (Exp 648).

### Energy-Time-Accuracy Tradeoffs in Thermodynamic Computing
- **Paper:** arXiv 2601.04358 (January 2026)
- **What:** Derives fundamental bounds on the energy-delay-deficiency (EDD) product for stochastic computation — the thermodynamic cost of achieving accuracy epsilon with energy E in time t. Key result: EDD >= kT*ln(2)/epsilon for Boltzmann sampling. Also derives control strategies (annealing schedules) that approach the EDD bound without needing the target distribution a priori.
- **Relevance to Carnot:** Carnot's SamplerBackend targets FPGA and TSU hardware. This paper provides the theoretical calibration signal: how much energy does a 100-spin Ising constraint check ACTUALLY cost at epsilon=0.01 accuracy? The EDD bounds can be measured empirically on KV260 (once Vivado synthesis completes) and compared to theoretical minimum. Also: the control strategies derived in this paper for approaching the EDD bound are directly applicable to Carnot's simulated annealing temperature schedules.
- **Concrete experiment:** Exp 650 is targeting RETRO-057 (LowRankKAEM accuracy). File arXiv 2601.04358 for hardware calibration milestone after KV260 bitfile synthesis succeeds. Implement EnergyTimeAccuracyProfiler as a SamplerBackend extension that measures the EDD product across different annealing schedules.
- **When to incorporate:** Milestone 2026.04.50+ — after KV260 bitfile synthesis succeeds (human must install Vivado). File now for reference.

### Accelerated Speculative Decoding via Sparse Verification (SSDV)
- **Paper:** arXiv 2512.21911 (December 2025)
- **What:** SSDV reduces verification latency in speculative decoding by skipping attention and FFN computations for tokens that are unlikely to be rejected, using a lightweight acceptance predictor. Achieves 40-60% verification overhead reduction with <1% quality degradation.
- **Relevance to Carnot:** Carnot's HERMES v2 live generation loop (Exp 641) runs SymCodeVerifier at every sentence boundary — expensive if done naively. SSDV's selective verification insight applies: skip verification for sentences with low violation probability (predicted by a lightweight acceptance predictor), run full SymCodeVerifier only on high-risk sentences. This could reduce HERMES v2 latency by 2-3x while preserving recall. File for after HERMES v2 recall baseline is established.
- **When to incorporate:** Milestone 2026.04.50+ — after HERMES v2 (Exp 641) establishes the recall-latency tradeoff baseline.

## 2026-04-21 arxiv Scan (Milestone 2026.04.48 Planning)

### HERMES — Multi-Module Tool-Augmented Verification for LLM Reasoning
- **Paper:** arXiv 2511.18760 (November 2025)
- **What:** Multi-modular tool-augmented agent that integrates formal verification into LLM reasoning. Four modules: LLM generates step → translator formalizes to Lean → prover verifies/counter-proves → feedback module signals next step. Memory block ensures proof continuity across steps.
- **Results:** 67% accuracy improvement on AIME'25, 80% fewer inference FLOPs. The feedback loop runs verifiers only on critical steps, not every token.
- **Relevance to Carnot:** This is a practical implementation of interwhen-style step verification. HERMES's architecture maps directly to Carnot: LLM generates CoT step → SymCodeVerifier verifies arithmetic → feedback injected before next step is generated. The prover module can be SymCodeVerifier (executable Python) instead of Lean (lighter weight, no formal logic dependency). Key advantage over interwhen: HERMES runs verification asynchronously at step boundaries, not every N tokens — much lower overhead for arithmetic checking.
- **Concrete experiment:** Exp 633 (HermesVerifierAdapter): adapt HERMES architecture for Carnot — SymCodeVerifier as the prover module, arithmetic claims from LLMAsExtractorV1 as the translator. Measure step-level violation recall on 25 known-incorrect responses.
- **When to incorporate:** Milestone 2026.04.48 — Phase 5 new research (Exp 633).

### AdapTrack — Constrained Decoding with Adaptive Backtracking
- **Paper:** arXiv 2510.17376 (October 2025)
- **What:** Constrained decoding that adaptively backtracks based on the fraction of invalid options at each step. When most next-token choices violate a constraint, backtrack to the last valid state rather than forcing the generation through an invalid branch. Mathematically proves the output distribution is identical to the model's own distribution under constraints — no output-intent distortion.
- **Results:** Up to 360% accuracy improvement on API generation tasks. The backtracking is proportional to invalidated probability mass, not fixed.
- **Relevance to Carnot:** After interwhen or SymCodeVerifier detects a mid-generation arithmetic violation, we need a repair mechanism. Currently Carnot uses post-hoc repair (full regeneration). AdapTrack offers in-generation repair: when SymCodeVerifier detects an arithmetic mismatch at step k, backtrack to step k-1 and regenerate with a constraint hint. The "fraction of invalid options" heuristic maps to SymCodeVerifier's violation detection confidence.
- **Concrete experiment:** Exp 635 (AdapTrack Constrained Generation): integrate AdapTrack backtracking with SymCodeVerifier. When violation detected mid-generation, backtrack N tokens and inject a correction hint into the prompt. Compare repair success rate vs post-hoc VerifyRepairPipeline.
- **When to incorporate:** Milestone 2026.04.48 — Phase 5 new research (Exp 635).

### Multilevel Training for KANs — Orders-of-Magnitude Accuracy Improvement
- **Paper:** arXiv 2603.04827 (March 2026)
- **What:** Multilevel training framework for spline-based Kolmogorov-Arnold Networks. Trains a sequence of KANs at increasing knot resolution using analytic geometric interpolation operators between levels. Exploits KAN structure to define natural coarse-to-fine refinement schedules. Achieves orders-of-magnitude accuracy improvement over standard training for comparable parameter count.
- **Results:** Minimax-optimal convergence rate O(n^(-2r/(2r+1))) for Sobolev functions. Numerical experiments show dramatic improvement particularly for physics-informed neural networks.
- **Relevance to Carnot:** KAEMEnergy (Exp 447) uses per-variable splines for exact sampling. Current training starts at K=256 knots directly. Multilevel training would start at K=16, converge, then refine to K=32, K=64, K=128, K=256 — each level initialized from the previous via interpolation. This should dramatically reduce training time and improve energy accuracy, directly fixing the RETRO-057 LowRankKAEM accuracy gap.
- **Concrete experiment:** Exp 634 (Multilevel KANs for KAEMEnergy): implement multilevel training for UnivariateKAEMLayer with K=16→32→64→128 refinement. Compare energy accuracy and training epochs vs current single-level training. Target: energy accuracy within 1% (vs current 5% gap in RETRO-057).
- **When to incorporate:** Milestone 2026.04.48 — Phase 5 new research (Exp 634). Also applicable to KAN energy tier in core Ising-KAN pipeline.

### Hidden Correctness via Symbolic Verification — Causal Reasoning in LLMs
- **Paper:** arXiv 2601.21210 (January 2026)
- **What:** Symbolic verification framework for uncovering hidden correctness in LLM causal reasoning. Uses symbolic execution to check whether LLM-generated reasoning chains are causally sound, even when the surface text looks confident. Reveals cases where LLMs reach correct final answers via invalid reasoning chains.
- **Relevance to Carnot:** Carnot's SymCodeVerifier currently catches arithmetic errors (47+28=76). This paper motivates extending verification to CAUSAL correctness: does step k causally entail step k+1? A chain can be arithmetically correct but logically incoherent. Relevant to the global consistency checker (Exp 172, 100% detection) — extend it with symbolic causal checking.
- **Concrete experiment:** File for Milestone 2026.04.49 — after SymCodeVerifier is deployed in live generation. Causal checker adds a second verification layer beyond arithmetic.
- **When to incorporate:** Milestone 2026.04.49+

## 2026-04-21 arxiv Scan (Milestone 2026.04.47 Planning)

### TRUST Agents — Multi-Agent LLM Claim Extraction (LLM-as-Extractor Architecture)
- **Paper:** arXiv 2604.12184 (April 2026)
- **What:** Collaborative multi-agent framework for fake-news detection and claim verification. Agent 1 uses NER to find numeric entities; Agent 2 forms arithmetic claims from those entities; Agent 3 verifies each claim via execution or lookup. Three-stage pipeline separates extraction from verification.
- **Relevance to Carnot:** Directly implements Goal #1b (LLM-as-extractor). The three-agent structure maps cleanly onto Carnot's pipeline: extraction LLM call → eval() verification → Ising energy scoring. Prior CoACEV4 failure (recall=4%) used a single-pass extraction prompt. TRUST Agents' multi-stage approach with an entity-finding agent first could recover more arithmetic claims from prose CoT.
- **Concrete experiment:** Exp 623 (TRUST Agents comparison): implement three-stage extraction — Stage 1: Qwen3.5-0.8B NER prompt to find all numeric entities in the CoT; Stage 2: construct arithmetic claims from entity pairs; Stage 3: verify via eval(). Compare recall vs LLMAsExtractorV1 (Exp 616) on same 25 incorrect live responses.
- **When to incorporate:** Milestone 2026.04.47 — Phase 4 (after LLMAsExtractorV1 baseline is established).

### AquaForte — LLM-Guided Quantified SMT Solving (Symbolic Arithmetic Extractor)
- **Paper:** arXiv 2601.04675 (January 2026)
- **What:** AquaForte uses LLMs to provide semantic guidance for quantified SMT constraint instantiation over uninterpreted functions. The LLM identifies the constraint structure; Z3 verifies the formal satisfiability. Combines LLM flexibility with SMT soundness guarantees.
- **Relevance to Carnot:** Carnot already has Z3ArithmeticExtractor (Exp 204) that failed on IT-model outputs. AquaForte's insight: use the LLM to GUIDE the SMT instantiation, not to do all extraction. Instead of asking the LLM "extract arithmetic claims," ask it "identify which Z3 variables correspond to the numbers in this reasoning step." This bridges the flexibility of LLM extraction with the soundness of Z3 verification.
- **Concrete experiment:** File for milestone 2026.04.47 Phase 1 as fallback if LLMAsExtractorV1 fails. LLM-guided Z3: use Qwen3.5-0.8B to identify variable bindings, then verify with Z3. No regex.
- **When to incorporate:** Milestone 2026.04.47 — alternative extractor path if LLMAsExtractorV1 (Exp 616) does not reach recall >= 20%.

### SymCode — Neurosymbolic Arithmetic Verification via Code Generation
- **Paper:** arXiv 2510.25975 (October 2025)
- **What:** SymCode translates arithmetic reasoning steps into executable Python code, then validates via execution. LLM generates code FROM the reasoning step; Python executes the code; output is compared to stated answer. Deterministic verification: no pattern matching, no symbolic parsing — execution is the ground truth.
- **Relevance to Carnot:** This is the cleanest realization of Goal #1b in the literature. The core insight: instead of extracting arithmetic CLAIMS from text, ask the LLM to WRITE CODE that computes the answer, then run it. For GSM8K, this means: (1) prompt Qwen3.5-0.8B with "write Python that computes [answer_step]"; (2) exec the code; (3) compare to stated answer. If mismatch: violation. No regex. No Z3. Pure execution.
- **Concrete experiment:** Incorporate as alternative to LLMAsExtractorV1 in Exp 616. SymCode-style code generation could be the most robust LLM-as-extractor approach because code execution is unambiguous. Target: recall >= 20% where pattern matching stuck at 4%.
- **When to incorporate:** Milestone 2026.04.47 — Phase 1 LLMAsExtractorV1 implementation (Exp 616). Include SymCode-style code generation as one of the extraction strategies alongside structured claim prompts.

### interwhen — Generalizable Intermediate Verification Framework
- **Paper:** arXiv 2602.11202 (February 2026)
- **What:** interwhen improves accuracy by up to 15% over standard CoT by inserting verification steps at intermediate points during generation. A monitor checks each reasoning step before the next is generated; if violation detected, the generation backtracks. Works across arithmetic, symbolic, and logical reasoning.
- **Relevance to Carnot:** This is the mid-generation verification architecture Carnot has been building toward. interwhen's monitor is exactly what Carnot's DSVD+LLMAsExtractor pipeline should do: at each CoT step boundary, run the extractor on the partial output and flag violations before proceeding. With LLMAsExtractorV1 achieving recall >= 20%, interwhen-style monitoring becomes practical. The 15% accuracy improvement claim motivates this as a Tier B product.
- **Concrete experiment:** File for Exp 627 (Milestone 2026.04.47 research frontier): implement interwhen-style intermediate verification using LLMAsExtractorV1. At each 32-token step boundary during Qwen3.5-0.8B generation, run the extraction LLM on the partial CoT and flag violations. If violation: emit a repair prompt before continuing. Compare end violation rate vs baseline.
- **When to incorporate:** Milestone 2026.04.47 — Phase 5 research frontier (after LLMAsExtractorV1 baseline confirmed).

### ORACLE — Constraint-Led Synthetic Data for Fine-Grained Verification
- **Paper:** arXiv 2603.21140 (March 2026)
- **What:** ORACLE generates fine-grained step-level verification training data using constraint-led elicitation. Symbolic reasoning verification validates each arithmetic operation. Outperforms binary correct/incorrect labeling by providing per-step supervision signal with constraint attribution.
- **Relevance to Carnot:** After LLMAsExtractorV1 achieves recall >= 20%, the next bottleneck is training the JEPA predictor on these extraction labels. ORACLE's constraint-led data generation could replace the current synthetic FOVER corpus with real-extraction-labeled data: each reasoning step gets a constraint label derived from LLMAsExtractorV1's output. This closes the offline/live gap (RETRO-066/068) because the training data is derived from the same extractor used at inference.
- **When to incorporate:** Milestone 2026.04.47+ — after LLMAsExtractorV1 produces labeled data; use ORACLE-style constraint labeling to build FOVER corpus v5 for JEPA v14 training.

## 2026-04-20 arxiv Scan (Milestone 2026.04.46 Planning)

### GenPRM — Generative Process Reward Model via Reasoning (LLM-as-Extractor Path)
- **Paper:** arXiv 2504.00891 (April 2025)
- **What:** A generative process verifier that performs explicit chain-of-thought reasoning at each step before producing a correctness verdict. Uses code verification as a tool. Outperforms discriminative PRMs on process supervision tasks.
- **Relevance to Carnot:** This is EXACTLY Goal #1b from research-program.md: "LLM-as-extractor". Instead of hand-engineered patterns (CoACE v1-v3), GenPRM's approach: (1) use a small LLM to extract verifiable arithmetic claims from the CoT step-by-step, (2) verify each claim via execution. The generative approach bridges the offline/live distribution gap because it reasons about the structure of the CoT rather than pattern-matching. CRITICAL FIX for RETRO-068 (CoACEV3 recall=4% on live data).
- **Concrete experiment:** Exp 603 (CoACEV4): implement GenPRM-style extraction — use a small Qwen3.5-0.8B call to identify arithmetic claims in each CoT step, then verify via Python eval(). Compare TP rate vs CoACEV3's 4% on the same live corpus. Target: TP rate >= 20%.
- **When to incorporate:** Milestone 2026.04.46 — Phase 1 CoACEV4 (Exp 603). CRITICAL PATH for RETRO-068/066.

### Streaming Hallucination Detection in Long CoT Reasoning
- **Paper:** arXiv 2601.02170 (January 2026)
- **What:** Models hallucinations as evolving latent states in long reasoning chains. Enables real-time prefix-level detection — catches the hallucination AS it forms, not post-hoc. Uses sequential state estimation on the LLM's hidden state sequence.
- **Relevance to Carnot:** Directly complements DSVD (arXiv 2503.03149). DSVD detects violation boundaries via hidden-state probing; this paper shows how to model the temporal dynamics of hallucination formation across a CoT. Together: DSVD finds the boundary, streaming detection confirms the trajectory. Could improve DSVD fine-tuning by providing richer temporal supervision signal (not just "violated at token T" but "violation building from token T-N to T").
- **Concrete experiment:** Exp 604 (DSVD Live Fine-Tuning): incorporate temporal state tracking when fine-tuning DSVDAdapter on live pairs. The streaming signal provides more training data per response (N state transitions instead of 1 boundary label).
- **When to incorporate:** Milestone 2026.04.46 — Phase 1 DSVD fine-tuning (Exp 604).

### CAPO — Calibration-Aware Policy Optimization for Reasoning LLMs
- **Paper:** arXiv 2604.12632 (April 2026)
- **What:** CAPO jointly optimizes accuracy and calibration via a logistic AUC surrogate loss during RLVR. Produces LLMs that are both accurate AND well-calibrated — their confidence scores actually reflect true probability of correctness. Works with contrastive training pairs.
- **Relevance to Carnot:** JEPA v12 AUC=1.0 suggests it may be outputting extreme confidence scores (near 0 or 1) rather than calibrated probabilities. CAPO's calibration loss directly addresses this: instead of optimizing only for correct/incorrect discrimination (the current CPMI margin loss), also penalize overconfidence. Apply to JEPA v13 if v12 OOD validation fails, or to NUP Probe v6 training.
- **Concrete experiment:** Exp 608 (NUP Probe v6): add CAPO calibration loss alongside contrastive margin loss when training on live corpus. Target: NUP v6 AUC >= 0.80 WITH calibration (confidence 0.80 means 80% of predictions correct, not just AUC).
- **When to incorporate:** Milestone 2026.04.46 — Phase 2 NUP Probe v6 (Exp 608). Also applicable to JEPA v13 if needed.

### Adaptive Constraint Propagation via Meta-Reinforcement Learning
- **Paper:** arXiv 2601.00095 (January 2026)
- **What:** MetaJuLS learns universal constraint propagation policies that adapt across tasks and input distributions via meta-RL. 1.5-2.0x speedup over task-specific baselines. Enables online adaptation of constraint reasoning without full retraining.
- **Relevance to Carnot:** Self-Learning Tier 2 (Constraint Memory) — instead of static constraint templates learned from a fixed training set, MetaJuLS-style meta-RL could enable CoACEV4 to adapt its extraction policy from new live pairs during inference, without retraining. This is the "online adaptation" capability the research-program.md prescribes.
- **Concrete experiment:** File for .47 — after CoACEV4 baseline is established, add meta-RL adaptation loop. The adaptation policy updates CoACE's pattern weights from each live inference batch.
- **When to incorporate:** Milestone 2026.04.47+ — after CoACEV4 shows baseline recall >= 20%.

## 2026-04-20 arxiv Scan (Milestone 2026.04.45 Planning)

### OTV — One-Token Verification for Reasoning Correctness (Ultra-Lightweight Verifier)
- **Paper:** arXiv 2603.01025 (March 2026)
- **What:** Adds a single learnable verification token via LoRA to any LLM to estimate reasoning correctness in one forward pass — no separate verifier model, no rollouts. Achieves performance competitive with full PRM at 90% lower token cost.
- **Relevance to Carnot:** Carnot's EORM (55M params, Tier 2) is expensive for real-time beam search. OTV offers a near-zero-cost verifier head attached to the existing Qwen3.5-0.8B or Gemma4-E4B-it model — no separate inference call. If OTV achieves comparable AUC to EORM on the FOVER corpus, it replaces EORM as the default Tier 2, cutting cascade latency by ~10ms per check. The LoRA adapter is hardware-portable (dot product over frozen weights + one extra vector).
- **Concrete experiment:** Attach OTV LoRA head to Qwen3.5-0.8B, train on 100 live FOVER pairs (from Exp 578), compare AUC vs EORM on held-out FOVER test set. If OTV AUC >= EORM AUC - 0.05, recommend OTV as default Tier 2.
- **When to incorporate:** Milestone 2026.04.45 — Phase 1 or 3, combine with DSVD live validation (Exp 592) or JEPAv12 retrain (Exp 593).

### PROGRS — Outcome-Conditioned PRM Centering (Prevents Reward Hacking in JEPA Training)
- **Paper:** arXiv 2604.02341 (April 2026)
- **What:** Introduces outcome-conditioned centering to shift PRM scores of incorrect trajectories to zero mean within each prompt group, safely integrating process rewards into GRPO without reward hacking. Improves math reasoning with fewer rollouts.
- **Relevance to Carnot:** JEPA v11 trained with CPMI contrastive loss on 9 synthetic pairs achieved AUC=1.0 — likely overfitting. PROGRS outcome-conditioned centering could prevent the same overfitting when retraining on 100 live pairs (Exp 593/JEPA v12). The centering ensures that the energy gap between correct and incorrect chains is calibrated against the distribution of errors for that prompt group, not just against a fixed margin. This directly addresses the hedging-to-0.5 failure mode that plagued v8/v9/v10.
- **Concrete experiment:** Apply PROGRS outcome-centering to JEPA v12 training in Exp 593. Instead of fixed hinge margin, use group-normalized energy gap: E_gap_i = (E_incorrect_i - E_correct_i) / std(E_gap for all pairs in prompt_group_i). Target: AUC >= 0.75 with stable training (no regression across epochs).
- **When to incorporate:** Milestone 2026.04.45 — JEPA v12 retrain (Exp 593). CRITICAL PATH for RETRO-063 validated closure.

### FACT-E — Causality-Inspired Evaluation for Trustworthy CoT Reasoning
- **Paper:** arXiv 2604.10693 (April 2026)
- **What:** Uses controlled perturbations as instrumental variables to measure genuine logical step-to-step dependence in CoT, separating real causal entailment from model-bias artifacts. Provides more reliable faithfulness estimates for chain-of-thought trajectory selection.
- **Relevance to Carnot:** CoACEExtractorV3 detects arithmetic errors via pattern matching. FACT-E's causal faithfulness score could serve as an ADDITIONAL energy signal: steps that are causally disconnected from prior steps (even if arithmetically correct) should have higher energy. This extends Carnot's verifier from "arithmetic correctness" to "logical faithfulness" — catching errors where the numbers add up but the reasoning chain is non-sequitur.
- **Concrete experiment:** Add `FACT-E causal faithfulness probe` as auxiliary energy term in CoACE v3 (Exp 591): after extracting arithmetic violations, compute causal disconnection score for each step transition. If score > threshold, flag as a faithfulness violation even if arithmetic is correct.
- **When to incorporate:** Milestone 2026.04.45 — incorporate as optional feature in CoACE v3 (Exp 591). Do not make it required for the gate (keep gate on arithmetic recall).

### p-bit Synchronous Ising Machine Architecture (FPGA Cost Reduction)
- **Paper:** arXiv 2604.01564 (April 2026)
- **What:** Shows synchronous p-bit Ising machine architectures achieve comparable solution quality at less than half the hardware cost of asynchronous designs, with low-resolution DACs and structured digital control preserving correct annealing dynamics.
- **Relevance to Carnot:** Directly relevant to KV260 FPGA Ising machine work (Exps 584/585). The synchronous architecture described is implementable on Zynq fabric (KV260 uses Zynq UltraScale+). Current `ising_sampler_v1.v` uses a pseudo-asynchronous update schedule — switching to synchronous-DAC-free design reduces resource utilization by ~50%, potentially enabling 2x more spins in the same FPGA area.
- **Concrete experiment:** After Vivado installation: synthesize both synchronous and asynchronous variants of the Ising sampler, compare LUT utilization and sampling throughput.
- **When to incorporate:** After Vivado installation (human action required). File for the milestone where KV260 bitfile synthesis succeeds.

### EBM Dynamical Models Tutorial — Phase 3 Architecture Reference
- **Paper:** arXiv 2604.05042 (April 2026)
- **What:** Comprehensive tutorial connecting continuous-time Hopfield networks, Boltzmann machines, dense associative memories, oscillator-based optimization, and proximal-descent dynamics under a unified control-theoretic energy landscape framework.
- **Relevance to Carnot:** Phase 3 foundation model architecture. The proximal-descent dynamics connect Carnot's Ising repair loop to classical Hopfield dynamics, providing theoretical grounding for the three-stage Prelude→IsingRepairLoop→Coda architecture. The oscillator-based optimization section validates the FPGA/thermodynamic hardware path via physical Ising machine theory.
- **When to incorporate:** Phase 3 architecture work — use as reference when designing the Kona bridge (python/carnot/phase3/).

## 2026-04-20 arxiv Scan (Milestone 2026.04.44 Planning)

### CPMI — Contrastive Pointwise Mutual Information for Process Reward Models (Direct Fix for RETRO-063)
- **Paper:** arXiv 2604.10660 (April 2026)
- **What:** Proposes CPMI (Contrastive Pointwise Mutual Information) as an automatic step-level reward labeler. Instead of binary correct/incorrect labels, CPMI measures how much each reasoning step increases mutual information between that step and the correct final answer, relative to hard-negative wrong answers. Reduces annotation time by 84%, token generation by 98%. Outperforms Monte Carlo estimation. Hard-negative mining ensures contrastive pairs have genuine energy gaps.
- **Relevance to Carnot:** RETRO-063: JEPA predictor AUC stuck at 0.44 (anti-correlated) despite PURE objective. The root cause is that binary BCE labels allow hedging to P=0.5. CPMI explicitly constructs contrastive (correct_chain, incorrect_chain) pairs via hard-negative mining — the same mechanism that made NUP Probe v4 achieve AUC=1.0. Directly applicable as the training objective for JEPA v11 in Exp 577 (pair builder) and Exp 580 (retrain).
- **Concrete experiment:** Exp 577: JEPACPMIContrastivePairBuilder — use CPMI-style hard-negative mining to construct (correct_chain, incorrect_chain) training pairs from FOVER corpus. For each incorrect response, find the HARDEST incorrect step (highest MI with wrong answer = most misleading step). Use this as the explicit negative in contrastive margin loss. Target JEPA v11 AUC >= 0.600.
- **When to incorporate:** Milestone 2026.04.44 — Phase 1 JEPA pair builder (Exp 577). CRITICAL PATH for RETRO-063.

### DSVD — Dynamic Self-Verify Decoding (Parallel Verification + Rollback)
- **Paper:** arXiv 2503.03149 (March 2025, EMNLP 2025)
- **What:** Real-time hallucination detection via parallel self-verification and dynamic rollback during generation. A specialized hallucination detector analyzes LLM internal states without additional text generation cycles. When violation detected, uses hidden state rollback to correct the specific problematic tokens rather than regenerating the full output. Demonstrated on factual QA and arithmetic reasoning.
- **Relevance to Carnot:** Carnot's current pipeline is purely post-hoc: generate full response → extract → verify → repair. DSVD shows that verification during generation (not after) is practical. Dynamic rollback maps to Carnot's repair-via-KAN energy — instead of regenerating the full response, identify the specific arithmetic step that went wrong and patch it. Directly addresses the repair precision gap in Exp 569 (7 repairs, only 1 improved = 14% repair success rate).
- **Concrete experiment:** Exp 587: DSVDAdapter — implement a lightweight verification head on Qwen3.5-0.8B hidden states that predicts arithmetic violation probability at each step boundary (every 32 tokens). Compare violation detection AUC vs CoACEExtractor v2 (post-hoc). Target: mid-generation detection at AUC > 0.60. CPU prototype (no GPU needed for hidden-state probe design).
- **When to incorporate:** Milestone 2026.04.44 — Phase 6 new research (Exp 587).

### CPMI Calibrated Hindsight — MISE Framework (Reward Signal Calibration)
- **Paper:** arXiv 2604.11611 (April 2026)
- **What:** MISE (Mutual Information Self-Evaluation) framework for calibrating hindsight rewards. Uses generative self-evaluation as dense reward signals, calibrated against sparse environmental feedback. Proves equivalence to minimizing MI + KL divergence between policy and proxy. Enables bootstrapping dense internal rewards from sparse external verification signals.
- **Relevance to Carnot:** After RETRO-033 (12 attempts, 0% improvement), the repair step has no feedback signal — the only reward is the final correctness check. MISE's dense reward calibration could provide step-level signals for repair optimization without requiring per-step annotation. Filed for .45 after RETRO-033 resolved.
- **When to incorporate:** Milestone 2026.04.45+ — after live verify-repair shows first positive result and produces repair trace data.

### Interleaved Formal-Logic Verification During Generation
- **Paper:** arXiv 2601.22642 (January 2026)
- **What:** Framework that dynamically interleaves formal symbolic verification during LLM generation, catching logical errors mid-chain rather than post-hoc. Two-stage: verification-guided SFT then policy optimization. Achieves 10.4-14.2% gains on math/logic reasoning.
- **Relevance to Carnot:** Current CoACE pipeline runs after full response generation. Interleaved verification could catch arithmetic violations mid-generation and redirect the response before committing to a wrong path. Complements DSVD (arXiv 2503.03149). Filed for .45+ after DSVD prototype establishes the mid-generation detection baseline.
- **When to incorporate:** Milestone 2026.04.45+ — after DSVD Exp 587 establishes feasibility.

### HISR — Hindsight Segmental Process Rewards for Multi-Turn RL
- **Paper:** arXiv 2603.18683 (March 2026)
- **What:** Segment-level process rewards (avoiding fine-grained turn-level noise) modulated by hindsight importance scores. Hindsight model reflects preference for actions given trajectory outcome. Modulation ratios measure action importance, improving credit assignment reliability for multi-turn agent tasks.
- **Relevance to Carnot:** Carnot's Tier 1 self-learning accumulates violations across batches but uses uniform credit assignment — every violation counted equally. HISR's hindsight modulation would weight constraint violations by how much they predicted the final incorrect outcome. This could dramatically improve ConstraintAdditionFromMemory's signal quality. Filed for .45 after FR-11 relay produces enough trace data.
- **When to incorporate:** Milestone 2026.04.45+ — after FR-11 real-violations relay accumulates 100+ traces.

### FLIP — Small Reward Models via Backward Inference
- **Paper:** arXiv 2602.13551 (February 2026)
- **What:** FLIP reformulates reward modeling as backward inference: given a response, infer the instruction or constraint that would produce it. Reference-free, rubric-free. Outperforms LLM-as-Judge by 79.6% on 4 domains. Works on 13 small LMs without requiring strong reasoning capability.
- **Relevance to Carnot:** Carnot's repair quality scoring is currently binary (correct/incorrect). FLIP's backward inference could score repair success by asking "what constraint was this response trying to satisfy?" — identifying whether the repair addressed the right violation. Filed for .45+ after repair pipeline produces enough (original, repair, verdict) triples.
- **When to incorporate:** Milestone 2026.04.45+ — after consistent repair data accumulates.

## 2026-04-20 arxiv Scan (Milestone 2026.04.43 Planning)

### Caco — Code-Execution Verification for Arithmetic CoT (Direct Fix for RETRO-061)
- **Paper:** arXiv 2510.04081 (October 2025)
- **What:** Caco (Code-Assisted Chain-of-Thought) synthesizes verifiable CoT by translating arithmetic reasoning steps into executable Python code and validating via execution. Built the Caco-1.3M dataset from code-grounded math problems. Key insight: regex and Z3 both fail on IT-model prose CoT because models write narrative equations ("We add 47 and 28 to get 76") not formal syntax. Code execution catches this: `eval("47+28") != 76` → violation detected.
- **Relevance to Carnot:** This is the direct fix for RETRO-061 (VeriCoTStepValidator TP=0 on live IT-model outputs). The current extractors fail because they check FORMAT (regex) or LOGIC (Z3 UNSAT) but not ARITHMETIC CORRECTNESS. Caco-style execution: (1) extract arithmetic expressions from CoT prose using a simple parser, (2) execute them as Python, (3) compare result to stated answer. No regex patterns. No Z3. Handles any arithmetic format IT models produce.
- **Concrete experiment:** Exp 564: CoACEExtractor (Code-Assisted Constraint Extraction) — implement an extractor that parses arithmetic equations from prose CoT, translates each to a Python `eval()` expression, and flags violations where `eval(lhs) != rhs`. Run diagnostic on Exp 554's 25 known-incorrect live responses. Target: TP rate > 0 (any improvement over VeriCoT's 0/25).
- **When to incorporate:** Milestone 2026.04.43 — Phase 1 extraction redesign (Exp 564). CRITICAL PATH.

### PURE — Min-Form PRM Objective as JEPA Training Signal (Direct Fix for RETRO-060)
- **Paper:** arXiv 2504.15275 (April 2025, NeurIPS 2025)
- **What:** "Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs." Solves reward hacking in PRMs by replacing sum-of-step-scores with the MINIMUM over all future step scores: score(prefix) = min(score(step_t), ..., score(step_T)). One bad step dominates the training signal, eliminating the common failure mode where summed scores allow models to game individual step evaluations. Achieves verifier-grade performance using only 30% of steps.
- **Relevance to Carnot:** JEPA predictor trained for two consecutive retrains (Exps 543, 557) both produced AUC < 0.5 with binary BCE loss. The binary label (correct/incorrect per step) allows the model to hedge toward 0.5 everywhere. The PURE min-form objective enforces a contrastive margin implicitly: the minimum step score in an incorrect reasoning chain forces the model to assign high violation energy to the worst step. This is a direct drop-in replacement for JEPA's binary CE loss: instead of `BCE(score, is_correct)`, use `min_form_loss(step_scores)` over the chain.
- **Concrete experiment:** Exp 566: JEPAPUREMinForm — replace JEPA's binary BCE loss with the PURE min-form objective. Train on 132-pair FOVER corpus v2. Target: AUC >= 0.700 (any value above 0.5 would be a breakthrough after two anti-correlated retrains).
- **When to incorporate:** Milestone 2026.04.43 — Phase 2 JEPA redesign (Exp 566). CRITICAL PATH.

### HalluField — Thermodynamic Energy-Path Hallucination Detection
- **Paper:** arXiv 2509.10753 (September 2025)
- **What:** Models LLM responses as token-path ensembles, assigns energy and entropy to each path from output logits using field-theoretic principles, flags hallucinations via thermodynamic instability (high partition function variance). No fine-tuning required — operates on logits directly. Reports strong AUROC on TruthfulQA and HaluEval.
- **Relevance to Carnot:** The verification cascade already has logit-based Tier 0 signals (SpilledEnergy 0b, NUP Probe 0c). HalluField adds a field-theoretic thermodynamic signal that is orthogonal to these: it characterizes the DISTRIBUTION of token paths rather than individual token entropy. This is a natural Tier 0e. The thermodynamic framing aligns directly with Carnot's EBM foundation — energy-path stability is what Carnot's energy function is meant to measure.
- **Concrete experiment:** Exp 571: HalluFieldTier0e — implement HalluField's partition function variance scorer over the logit distribution, add as Tier 0e to verification cascade. Benchmark AUC vs SpilledEnergy (0b) and NUP Probe v4 (0c) on 132-pair FOVER corpus. CPU-only (operates on logits).
- **When to incorporate:** Milestone 2026.04.43 — Phase 5 new research (Exp 571).

### Process Reward Agents (PRA) — EBM as Step-Level Reward Module
- **Paper:** arXiv 2604.09482 (April 2026)
- **What:** Decouples a frozen LLM policy from a step-wise reward module; uses beam search pruning per step guided by the reward module. Achieves 80.8% on MedQA with a 4B parameter model (+25.7% gain over greedy). Reward module is lightweight and model-agnostic — any scorer can be plugged in.
- **Relevance to Carnot:** Treats the EBM energy function as the PRA reward module — plug Carnot's Ising/KAN/EORM scorer into the PRA beam-search framework to steer LLM generation toward low-energy (constraint-satisfying) reasoning paths at inference time. This is a more principled version of the guided decoding work (Exp 110, Tier B product) with a proven framework. Especially compelling now that EORM + JEPA are being retrained on real data.
- **Concrete experiment:** Exp 572: PRAEBMBeamSearch — implement PRA beam search with EORM as the step-level reward module (K=3 candidates per step, EORM selects minimum energy). CPU prototype on 20 synthetic arithmetic problems. Compare violation rate vs greedy baseline.
- **When to incorporate:** Milestone 2026.04.43 — Phase 5 new research (Exp 572).

### Frequency-Aware Attention Hallucination Detection
- **Paper:** arXiv 2602.18145 (February 2026)
- **What:** Treats attention distributions as discrete frequency-domain signals; high-frequency attention energy (measured via DFT of attention weight rows) flags unstable token grounding correlated with hallucination. Lightweight (no retraining), works on any pre-trained LLM.
- **Relevance to Carnot:** Complementary to SinkProbe (Tier 1, arXiv 2604.10697). SinkProbe measures sink CONCENTRATION (low frequency: attention pooling to BOS/EOS tokens). This paper measures sink TURBULENCE (high frequency: attention scattered across many tokens). Together they provide a fuller characterization of attention geometry as hallucination signals. Could be added as a second feature to SinkProbe's scalar output to improve its AUC.
- **When to incorporate:** Milestone 2026.04.44+ — SinkProbe enhancement. Lower priority than Exps 564/566 critical path.

### Symbolic-KAN — KAN with Interpretable Discrete Symbolic Structure
- **Paper:** arXiv 2603.23854 (March 2026)
- **What:** Symbolic-KAN replaces KAN's continuous spline activations with discrete symbolic equations (sin, cos, polynomial, log, exp). Bridges symbolic regression and neural networks. The KAN learns which symbolic form best fits each activation, producing human-readable constraint rules.
- **Relevance to Carnot:** The current KAN energy tier (KAEMEnergy, Exp 447) uses opaque spline activations. Symbolic-KAN would make the energy function interpretable: "constraint fires when x > 0.7 OR y < 0.3" rather than an uninterpretable spline. This directly addresses the constraint extraction opacity — if we know the learned constraint RULES, we can explain why the pipeline fires on any given input.
- **When to incorporate:** Milestone 2026.04.44+ — KAN tier interpretability upgrade. Not critical path for .43.

## 2026-04-20 arxiv Scan (Milestone 2026.04.42 Planning)

### Energy-per-Token in LLM Inference — Hardware-Level Verification Metric
- **Paper:** arXiv 2603.20224 (March 2026)
- **What:** Advocates using hardware-measured energy-per-token (joules/token) as the primary efficiency metric for LLM inference, superseding throughput or FLOP counts. Shows that token-level energy varies significantly across generation steps — early tokens (attending to full context) cost 3-8x more than later tokens. Proposes energy-aware routing: route short-context queries to smaller models, long-context queries to larger ones.
- **Relevance to Carnot:** Carnot's per-step EORM energy scores are learned proxies for reasoning quality, not hardware energy. But the two correlate: hardware energy spikes during "hard" reasoning steps (long attention, large hidden state activations), which are also where Carnot's EORM energy is high. This opens a hardware-level validation path: compare Carnot's EORM energy scores against hardware power traces on the same generation. If they correlate, hardware energy becomes a free calibration signal for EORM training — no labeling needed, just a power meter.
- **Concrete experiment:** Record hardware power trace (via RAPL or nvml) during 25 live GSM8K generations. Compute correlation between Carnot EORM energy at each 32-token boundary and measured hardware watt-per-token. If r > 0.5, use hardware energy as EORM training signal. CPU-only (RAPL available on AMD Ryzen AI 9). Deliverable: correlation_coefficient + hardware_vs_eorm_scatter.
- **When to incorporate:** Milestone 2026.04.43+ — requires stable EORM on real data first (Exp 556). File as hardware calibration experiment.

### OpenMythos — Recurrent-Depth Transformer Architecture Reconstruction
- **Repo:** https://github.com/kyegomez/OpenMythos (Kye Gomez, community project; not affiliated with Anthropic)
- **What:** Theoretical PyTorch reconstruction of the Claude Mythos architecture from published research — a looped Recurrent-Depth Transformer (RDT) with three stages: Prelude (standard transformer) → Recurrent Block (looped T times, hidden state evolves as `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)`) → Coda (standard transformer). Uses switchable MLA/GQA attention, sparse MoE feedforward, and enforces stability via `ρ(A) < 1` spectral radius constraint on the linear injection matrix. Configurable from 1B to 1T parameters with a FineWeb-Edu training script. **Ships architecture code only — no trained weights, no benchmarks, no empirical verification of behavior.**
- **Relevance to Carnot — four distinct angles:**
    1. **Validation-moat evidence.** A full architectural clone exists publicly without reproducing Claude's actual reasoning behavior. This is the clearest worked example of the thesis underpinning Carnot's "Why Carnot" README section (the Decrypt/Vidoc framing): architecture is commoditized, verification is the moat. Worth citing directly.
    2. **Structural near-isomorphism to EBM descent.** Mythos's update rule `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` is a learned recurrent update on continuous latent state. Carnot's Ising repair loop is `h_{t+1} = h_t − η·∇E(h_t, e)` — a gradient-flow dynamical system on an energy surface. Both are discrete-time recurrent dynamical systems with an input-injection term; Mythos learns `(A, B, Transformer)`, Carnot learns `E`. They converge on the same "test-time-compute independent of model size" lever from opposite directions. This makes Mythos a legitimate comparison baseline for any Phase-3 foundation-model work.
    3. **Phase-3 blueprint validated.** The three-stage **Prelude → Recurrent Block → Coda** shape is what Carnot's EBM foundation model should be: `embed → (Ising/JEPA repair loop, T iterations) → decode`. A Carnot variant replaces the looped Transformer with an energy-descent step that has a *stronger* stability guarantee — CD-trained EBMs are bounded below, so iterates cannot diverge without needing explicit `ρ(A) < 1` parameter clipping.
    4. **Concrete borrowings.** (a) Muon + AdamW optimizer pairing — Muon's orthogonality guarantees may stabilize the KAN low-rank fast-path where Exp 544's SVD approximation produced 99% energy error. Try as a one-line optimizer swap in .43+. (b) Sparse MoE routing (routed + shared experts) as an alternative scaling path to dense Boltzmann — cheaper inference at similar capacity. Queue as an architectural variant experiment.
- **Concrete experiment (Phase-3 prototype):** Implement `EBMMythosPrototype`: three-stage `Prelude(small MLP) → IsingRepairLoop(T=8 iterations, energy-descent update) → Coda(small MLP)` on a toy arithmetic task (addition mod 100). Verify (a) monotone energy decrease across the T loops, (b) accuracy improves with T, (c) stability holds without any `ρ(A)` clipping. CPU-only prototype. Deliverable: accuracy vs T curve + energy monotonicity plot.
- **When to incorporate:** Milestone 2026.04.43+ — Phase 3 exploratory architecture. Queue after the .42 live-data sprint produces real CoT pairs; prototype the 3-stage shape on a toy task first before scaling. File OpenMythos's repo + the "architecture-clone, no empirical replication" framing in `README.md` "Why Carnot: the validation moat" section to strengthen the public narrative.

## 2026-04-19 arxiv Scan (Milestone 2026.04.41 Planning)

### EBM Calibration of Latent Chain-of-Thought — Energy-Guided Implicit Reasoning
- **Paper:** arXiv 2511.07124 (November 2025)
- **What:** Integrates a small EBM to calibrate latent thought tokens during implicit CoT generation. The EBM assigns an energy scalar to each latent reasoning step and guides sampling toward lower-energy (more coherent) trajectories. Achieves multi-CoT-level accuracy with a single forward pass on LLaMA-3.1-8B.
- **Relevance to Carnot:** Direct blueprint for online energy-guided verification at the step level — not just final output scoring. The EBM scalar acts as a constraint signal over latent reasoning tokens. Compatible with Carnot's JEPA tier (Tier 3): predict constraint energy from partial CoT → steer generation away from violations before they materialize in the final response. Validates that per-step energy is both measurable and useful for coherence.
- **Concrete experiment:** Implement `LatentCoTEBMCalibrator`: wrap Qwen3.5-0.8B generation loop, compute EORM energy at each reasoning step boundary (every 32 tokens), apply soft temperature adjustment toward lower-energy continuations. Compare violation rate on 50 GSM8K questions vs uncalibrated baseline. CPU-only with synthetic energy function. Deliverable: per-step energy distribution + violation rate comparison.
- **When to incorporate:** Milestone 2026.04.41 — Phase 6 new research (Exp 545 alt).

### Efficient Test-Time Scaling via Internal-State Probing — 810x Smaller than PRM
- **Paper:** arXiv 2511.06209 (November 2025)
- **What:** Lightweight transformer probes on LLM internal hidden states (single linear layer per layer) match or exceed much larger Process Reward Models (PRMs) for step-level reasoning credibility estimation. Probes are 810x smaller than the process reward model baseline. Can be applied to any pre-existing LLM without modification.
- **Relevance to Carnot:** Carnot's EORM (55M params, Tier 2) is a large model for per-step scoring. An internal-state probe replaces EORM with a per-layer linear probe that reads directly from the LLM's residual stream — no separate model, no additional inference. If the probe achieves EORM-level AUC at 810x smaller, it should become the default Tier 2. The linear probe is FPGA-native (dot product over hidden states).
- **Concrete experiment:** Implement `InternalStateProbe(model, probe_layer=-4)`: extract hidden state at layer -4 from Qwen3.5-0.8B, project to scalar via learned linear layer, train on 57 real FOVER CoT pairs. Compare AUC vs EORM (55M params) on same test set. CPU-only (no new model — just probe weights). Deliverable: AUC comparison + parameter count ratio.
- **When to incorporate:** Milestone 2026.04.41 — Phase 6 new research (Exp 545).

## 2026-04-19 arxiv Scan (Milestone 2026.04.40 Planning)

### Adaptive Rectification Sampling — EORM as PRM for Test-Time Compute Scaling
- **Paper:** arXiv 2504.01317 (April 2025)
- **What:** Uses a process-supervised reward model (PRM) as a verifier for adaptive test-time compute scaling. At each step, the PRM scores multiple candidate continuations and selects the highest-reward one. Models can "rethink" at the step level when the PRM signals low reward. Achieves measurable gains on GSM8K and MATH500 without additional training.
- **Relevance to Carnot:** Carnot's EORM (Exp 346: learned energy reward model, 55M params) is structurally identical to the PRM in this paper. The pipeline: generate N candidates per step → score each with EORM → select minimum-energy candidate → proceed. This is a zero-infrastructure integration: EORM already exists, the only addition is N-candidate generation + EORM ranking. Expected: reduces per-step violation rate by selecting lower-energy continuations before committing.
- **Concrete experiment:** Exp 531: EORMAsTestTimePRM — generate K=3 candidates per question, score all with EORM, select minimum energy, compare against greedy baseline. CPU-only with synthetic data. Deliverable: comparison of violation rate greedy vs EORM-selected.
- **When to incorporate:** Milestone 2026.04.40 — Phase 4 (Exp 531).

### Potts Machine — Multi-Value Constraint States via Mean-Field
- **Paper:** arXiv 2602.04200 (February 2026)
- **What:** Restoring Sparsity in Potts Machines via Mean-Field Constraints. Potts machines are the q-state generalization of Ising machines (Ising = q=2, binary). Mean-field constraints maintain sparse coupling structure during optimization, enabling scalable Potts machines on parallel hardware. FPGA-native architecture demonstrated.
- **Relevance to Carnot:** Carnot's IsingEBM uses binary spin states (+1/-1) for constraint verification — each constraint is either satisfied or violated. Many real constraint states are multi-valued: correct / partially-correct / violated (q=3), or correct / underspecified / ambiguous / violated (q=4). A PottsMachineVerifier with q=3 could directly encode constraint confidence without binarizing. The mean-field approach maintains the FPGA-compatible sparse structure. This is a natural generalization of the Ising tier with a hardware path.
- **Concrete experiment:** Exp 534: PottsMachineVerifier — implement q=3 Potts sampler (Gibbs update over {-1,0,+1} spins), encode constraints as 3-state Potts couplings, benchmark AUROC vs binary IsingEBM on classification of correct/partial/violated constraint states.
- **When to incorporate:** Milestone 2026.04.40 — Phase 5 (Exp 534).

### GRPO Verifiable Rewards — Contrastive Loss as Verification Signal
- **Paper:** arXiv 2503.06639 (March 2025)
- **What:** Analyzes GRPO's effective loss for LLM reasoning under verifiable (binary) rewards. Shows that mean+variance reward calibration induces a contrastive loss structure: problems the model gets right and wrong are automatically paired as positive/negative. The verifiable reward (binary correct/incorrect) IS the contrastive signal, with no additional annotation needed.
- **Relevance to Carnot:** Carnot's NUP Probe v4 (Exp 523) established that contrastive training (maximize E(incorrect)-E(correct) gap) dramatically outperforms BCE (AUC=1.0 vs 0.40). GRPO's analysis explains WHY: the energy gap is the natural training objective for any binary-verifiable system. For Carnot, this means: accumulate (question, is_correct) pairs from live benchmarks → use GRPO-style contrastive pairing → retrain NUP Probe on these pairs without any additional labeling infrastructure. The live benchmark CoT pairs ARE the training signal.
- **When to incorporate:** Next JEPA/NUP retrain — use GRPO contrastive pairing as data construction strategy for Tier 1-3 self-learning models. File for milestone .40+.

### IR³ — Contrastive IRL for Reward Hacking Detection
- **Paper:** arXiv 2602.19416 (February 2026)
- **What:** Contrastive Inverse Reinforcement Learning reconstructs implicit reward functions by contrasting paired responses from post-alignment and baseline policies. Detects when a model is "gaming" the reward signal (reward hacking) by comparing response structures — reward-hacked responses have anomalously high reward but structurally diverge from the baseline policy.
- **Relevance to Carnot:** The VerifyRepairPipeline faces a similar problem: a model could produce a response that fools the ArithmeticExtractor by conforming to regex patterns without actually solving the problem correctly. IR³'s contrastive approach — score baseline vs pipeline response pairs — could detect when the repair process is introducing a different kind of error. File as a future quality-gating mechanism for the repair output.
- **When to incorporate:** Milestone .41+ — after live pipeline produces enough repair examples to analyze.

### AutoRefine — Continual Agent Refinement via Trajectory Distillation
- **Paper:** arXiv 2601.22758 (January 2026)
- **What:** AutoRefine converts agent interaction trajectories into reusable abstract strategic principles via offline self-distillation. During online interaction, the agent retrieves distilled principles to guide decisions. The lifecycle is: interact → distill trajectories into principles → store in principle repository → retrieve at inference time.
- **Relevance to Carnot:** This is exactly Tier 2 self-learning (ConstraintMemory). AutoRefine's "principle distillation" corresponds to Carnot's "constraint template generation from violation patterns." The offline distillation step (batch processing of accumulated violation pairs) maps to Carnot's end-of-session constraint consolidation. Key extension: AutoRefine's retrieval-at-inference adds a retrieval step Carnot doesn't have — look up stored constraint templates relevant to the current query before running Ising verification. This could reduce false positives by applying domain-specific constraints only when relevant.
- **When to incorporate:** Milestone .41+ — after FR-11 live relay is confirmed and real violation patterns have accumulated across multiple live sessions.

## 2026-04-19 arxiv Scan (Milestone 2026.04.39 Planning)

### Hallucination Basins — Dynamical Systems Framing of LLM Hallucination
- **Paper:** arXiv 2604.04743 (April 2026)
- **What:** Presents a geometric dynamical systems framework where hallucinations arise from task-dependent basin structures in latent space. Correct reasoning follows low-energy attractor basins; hallucinated reasoning drifts into shallow basins with high escape probability.
- **Relevance to Carnot:** Directly applicable to JEPA predictive verification. The basin depth (energy well depth) is a natural verification signal: a CoT prefix that is drifting toward a shallow basin is more likely to produce a violated constraint. JEPA predictors can learn to detect shallow-basin trajectories before the generation completes.
- **Concrete experiment:** Implement HallucinationBasinDetector: estimate latent-space basin depth from LLM hidden states at each generation step, compute escape probability, compare vs SpilledEnergy (Tier 0b) as a hallucination detection signal. Benchmark AUC on 200 synthetic CoT responses.
- **When to incorporate:** Milestone 2026.04.39 — Phase 5 new research (Exp 521).

### LeWorldModel — Stable End-to-End JEPA Training from Raw Observations
- **Paper:** arXiv 2603.19312 (March 2026)
- **What:** First JEPA trained stably end-to-end from raw observations using only two loss terms: (1) next-embedding prediction and (2) Gaussian latent regularization. 15M parameters, trainable on a single GPU, 48x faster planning than foundation models. Embedding space encodes genuine physical structure.
- **Relevance to Carnot:** JEPA predictor training has been unstable (AUC regression from 0.667 to 0.400 in Exp 472, recovered to 0.967 via curriculum in Exp 492). The two-term objective (prediction + Gaussian regularization) provides a principled regularization that prevents training collapse without needing curriculum scheduling. The Gaussian regularization is CPU-trivial to implement. Target: AUC >= 0.800 with stable training from session 1.
- **Concrete experiment:** Implement LeWorldModelJEPA with L_total = L_prediction + λ * L_regularization, where L_regularization = KL(q(z) || N(0,I)) (KL to standard Gaussian for each latent). Apply to CoT step embeddings. Compare training stability and AUC vs current curriculum approach.
- **When to incorporate:** Milestone 2026.04.39 — Exp 520 (LeWorldModel-JEPA stable training), and FR-11 retrain Exp 522.

### Constrained Decoding with Near-Zero Overhead — Schema Key Wording
- **Paper:** arXiv 2604.14862 (April 2026)
- **What:** Studies how constrained decoding enforces formal language constraints (JSON, XML) via DOMINO decoding and XGrammar engines with speculative decoding. Reviews how schema structure in key naming serves as an implicit instruction channel, reducing constraint violations without explicit enforcement.
- **Relevance to Carnot:** Energy-guided constrained generation (Exp 110, Tier B product). Carnot's VerifyRepairPipeline currently checks after generation; constrained decoding checks during generation at near-zero overhead. This paper shows that speculative decoding + grammar constraints can be energy-guided: the energy function penalizes constraint violations in real-time during generation.
- **Concrete experiment:** Integrate energy-guided constrained decoding: use DOMINO/XGrammar token mask + Carnot energy scoring to bias token selection toward constraint-satisfying continuations. Compare violation rate and quality vs post-hoc repair.
- **When to incorporate:** Milestone 2026.04.40+ — requires stable live benchmarks first.

### Low-Rank Energy Landscape — Logit Energy is Compressible
- **Paper:** arXiv 2604.04384 (April 2026)
- **What:** Demonstrates that logit energy fields in transformers reach 90% of their total variance in only 2-11 singular components (low-rank decomposition of the logit matrix). This means the energy landscape is inherently low-dimensional, not the high-dimensional tensor it appears to be.
- **Relevance to Carnot:** KAEMEnergy and KAN energy tiers currently operate on the full-dimensional energy space. If the energy landscape is low-rank (2-11 components), a rank-2 KAN is sufficient for 90% accuracy at 10-100x fewer parameters. Compute efficiency win: low-rank KAN energy computation is O(n * r) where r=11 instead of O(n * d) where d=hidden_size. This also informs JEPA predictor design: project to low-rank before predicting.
- **Concrete experiment:** Implement LowRankKAEMEnergy: compute SVD of logit matrix, project to top-k singular vectors (k=2,4,8,11), evaluate AUC vs full-rank KAEMEnergy. If k=11 achieves >95% of full-rank AUC, recommend as default. CPU-only.
- **When to incorporate:** Milestone 2026.04.39 — could replace Exp 521 if Hallucination Basins is lower priority. File for .39+.

## 2026-04-19 arxiv Scan (Milestone 2026.04.38 Planning)

### Semantic Energy — Boltzmann-Inspired Hallucination Detection from Logits
- **Paper:** arXiv 2508.14496 (August 2025)
- **What:** Introduces Semantic Energy as a replacement for semantic entropy in LLM uncertainty estimation. Combines semantic clustering with a Boltzmann-inspired energy distribution derived directly from logits (penultimate layer), capturing uncertainty cases where semantic entropy is overconfident. Achieves 13% average AUROC improvement over semantic entropy on hallucination detection benchmarks.
- **Relevance to Carnot:** SpilledEnergy (Tier 0b, arXiv 2602.18671) and NUP Probe (Tier 0c, arXiv 2603.19562) both operate at the sequence level. Semantic Energy operates at the cluster level using Boltzmann distributions — the energy function framework Carnot is built on. This is the most natural new Tier 0d: Boltzmann-cluster energy from logits, inserted between SpilledEnergy (token-level) and SinkProbe (attention-level). The Boltzmann formulation means it's differentiable and could be integrated into the EORM training objective.
- **Concrete experiment:** Implement BoltzmannSemanticEnergy (compute semantic cluster logit energies, Boltzmann-weight them, return hallucination score). Compare AUC vs SpilledEnergy on 200 live CoT pairs. Add as Tier 0d to verification cascade. Target: AUC > SpilledEnergy baseline.
- **When to incorporate:** Milestone 2026.04.38 — Phase 4 research (Exp 506).

### Cross-Layer Attention Probing (CLAP) — Residual Stream Hallucination Detection
- **Paper:** arXiv 2509.09700 (September 2025)
- **What:** CLAP constructs a sequence of tokens from LLM activations across the ENTIRE residual stream (all layers simultaneously), then applies attention over this cross-layer sequence to detect hallucinations. Processes information from early layers (surface syntax), middle layers (semantic integration), and late layers (generation decision) jointly. Captures inter-layer reasoning trajectories, not just final-layer features.
- **Relevance to Carnot:** NUP Probe v2 (Exp 496) failed to improve over v1 (AUC = 0.600, delta ~1e-16) because Bayesian semantic entropy averages over the sequence. CLAP's cross-layer joint attention is exactly the richer feature set needed: per-layer activations as a token sequence → attention over that sequence → hallucination score. NUP Probe v3 (RETRO-049) should implement CLAP-style cross-layer attention as its feature extractor.
- **Concrete experiment:** NUP Probe v3 (Exp 507): Add CLAPFeatureExtractor(model, layers=[-1,-4,-8,-12]) that constructs a (n_layers, n_tokens, hidden_size) tensor and applies multi-head attention to produce a scalar hallucination score. Retrain on real CoT pairs from Exps 502-503. Target AUC > 0.700 for Tier 0c promotion.
- **When to incorporate:** Milestone 2026.04.38 — NUP Probe v3 (Exp 507).

### Semantic Token Clustering — Efficient UQ for LLM Hallucination
- **Paper:** arXiv 2603.20161 (March 2026)
- **What:** Semantic Token Clustering (STC) groups token sequences into semantically consistent clusters using embedding clustering + prefix matching, then computes uncertainty within and between clusters. Dramatically reduces the number of LLM samples needed for UQ from 5-10 to 2-3 while maintaining AUROC quality.
- **Relevance to Carnot:** Carnot needs to evaluate multiple candidate responses (score_candidates MCP tool). STC's sample-efficient UQ could reduce the number of Gemma4 inference calls needed to estimate hallucination probability per question from O(5) to O(2). Direct integration: CarnotCandidateRanker could use STC to cluster candidate responses before scoring with the energy function — reducing inference budget by ~60%.
- **Concrete experiment:** Integrate STC into score_candidates MCP tool as an optional pre-clustering step. Compare: (a) score all 5 candidates independently vs (b) cluster first, score one per cluster. Measure accuracy vs cost tradeoff.
- **When to incorporate:** Milestone 2026.04.39 — production path for score_candidates optimization.

### Intrinsic-Energy JEPA — Quasimetric Representation Space for Reasoning
- **Paper:** arXiv 2602.12245 (February 2026)
- **What:** Intrinsic-Energy JEPA shows that JEPA-style predictive architectures naturally induce quasimetric spaces where distance encodes reachability. Value-guided JEPA planning: the embedding-space cost aligns with a goal-reaching value function, enabling sequential compositionality in reasoning chains. Key insight: distance in the learned embedding space is a proxy for reasoning difficulty, not just semantic similarity.
- **Relevance to Carnot:** JEPA predictor (Tier 3 self-learning) currently predicts constraint violation probability from partial CoT embeddings. If the embedding space is quasimetric, the JEPA predictor implicitly learns reasoning difficulty — which is exactly what we want: hard reasoning steps (high quasimetric distance from premise to conclusion) correlate with higher violation probability. Incorporate quasimetric regularization into JEPA training to make this explicit.
- **Concrete experiment:** Add quasimetric regularization term to JEPA Curriculum Retrain v4: L_total = L_prediction + λ * L_quasimetric, where L_quasimetric penalizes symmetry violations in the embedding distance (d(a,b) >> d(b,a) when a is a premise and b is a conclusion). Measure whether AUC and calibration improve.
- **When to incorporate:** Milestone 2026.04.38 — JEPA Live Retraining v4 (Exp 510).

### EB-JEPA Library — Open-Source Energy-Based JEPA Implementation
- **Paper/Repo:** arXiv 2602.03604 (February 2026)
- **What:** Lightweight open-source library for energy-based joint-embedding predictive architectures. Trains JEPA models that predict in representation space rather than pixel/token space. Integrates energy scoring into the JEPA objective directly: low energy = aligned (predicted, actual) embedding pair.
- **Relevance to Carnot:** Carnot's JEPA predictor (EORMModel + JEPAPredictor) was built from scratch. The EB-JEPA library provides a reference implementation of energy-based JEPA that could accelerate development. The energy-in-representation-space framing aligns exactly with Carnot's verification goal: predict whether (partial CoT, continuation) embedding pair will have high or low constraint energy.
- **When to incorporate:** Consider as a dependency when refactoring JEPA predictor for Phase 3 (Kona bridge). Not needed for Exp 510 retrain.

### AMD XDNA NPU for ML Training and Inference (arXiv 2504.03083)
- **Paper:** arXiv 2504.03083 (April 2025)
- **What:** First demonstration of LLM fine-tuning on AMD NPU using the IRON tool-flow. AMD Ryzen AI's XDNA architecture: spatial array of AI Engines in 2D grid. Achieved GPT-2 fine-tuning on NPU. XDNA 2 (newer generation) has higher throughput.
- **Relevance to Carnot:** NUP Probe v3 uses per-token entropy computation — a sequence of softmax operations over vocabulary (50k tokens). This is embarrassingly parallel and NPU-native. If NUP Probe's per-token entropy can run on the NPU while the LLM runs on GPU, the Tier 0c latency drops to near-zero (pipeline parallelism). Target: NPU probe latency < 5ms per token, matching LLM generation speed.
- **Concrete experiment:** Deploy NUP Probe v3 entropy computation on AMD NPU via IRON + ONNX Runtime VitisAI backend. Benchmark: NPU probe latency (ms/token) vs CPU probe latency. If NPU is 2x+ faster, recommend as the production Tier 0c path.
- **When to incorporate:** Milestone 2026.04.38 — AMD NPU experiment (Exp 511).

### CIKAN — Constraint Informed Kolmogorov-Arnold Networks
- **Paper:** arXiv 2412.03710 (December 2025)
- **What:** Constraint Informed KAN (CIKAN) for autonomous spacecraft rendezvous using Time Shift Governor. Integrates hard physics constraints directly into KAN spline structure. KAN's adaptive spline resolution naturally concentrates at constraint boundaries where the function is most complex.
- **Relevance to Carnot:** KAEMEnergy (Exp 447) uses univariate per-variable KAN splines for energy computation. CIKAN's constraint-informed spline initialization could improve KAEMEnergy's energy landscape for constrained verification problems: initialize splines near constraint boundaries with higher resolution, allowing the energy function to sharply distinguish near-boundary states. This is a direct improvement to the KAN fast-path tier.
- **When to incorporate:** Future KAN tier enhancement — not .38 priority. File for milestone .39+.

### Stochastic Ising Machine Sampling Advantage (arXiv 2504.18359)
- **Paper:** arXiv 2504.18359 (April 2025)
- **What:** Quantifies when stochastic Ising machines outperform standard Metropolis-Hastings for quantum simulations using neural-network quantum states. FPGA Ising machines demonstrated faster than conservative projections in controlled benchmarks.
- **Relevance to Carnot:** ParallelIsingSampler (Exp 290) achieves 183x speedup over thrml on CPU. The FPGA path (KV260, Exp 313) is still blocked on missing bitfile. This paper provides the theoretical performance envelope: if the Ising machine problem is a quantum simulation target, the sampling advantage can be analytically predicted. Use as the acceptance criterion for KV260 bring-up: only proceed if predicted advantage > 10x over ParallelIsingSampler.
- **When to incorporate:** KV260 bring-up session — validate expected advantage before bitfile synthesis.

## 2026-04-19 arxiv Scan (Milestone 2026.04.37 Planning)

### LLM-JEPA — Predictive Embedding Space for Stable Constraint Scoring
- **Paper:** arXiv 2509.14252 (2025)
- **What:** Applies JEPA predictive principles to LLM training by optimizing in embedding space (not token space). JEPA embeddings show superior robustness against distribution shift and consistent improvements on GSM8K, Spider, RottenTomatoes. The stable embedding space reduces variance across semantically equivalent inputs.
- **Relevance to Carnot:** JEPA embeddings could serve as better input representations for the EORM constraint scorer. JEPA's stability under distribution shift directly addresses the JEPA AUC regression problem (RETRO-040) — unstable embeddings cause unstable AUC. Test whether JEPA embeddings from LLM-JEPA improve EORM's AUROC on real CoT pairs.
- **Concrete experiment:** Compare EORM constraint satisfaction AUC on standard vs LLM-JEPA embeddings. If JEPA embeddings improve AUC stability, incorporate into JEPA Curriculum Retrain v3.
- **When to incorporate:** Milestone 2026.04.37 — JEPA recovery phase.

### Bayesian Semantic Entropy for Hallucination Detection
- **Paper:** arXiv 2603.22812 (AAAI 2026 oral)
- **What:** Hierarchical Bayesian framework for hallucination detection. Adaptively allocates sampling budget based on entropy uncertainty — high-uncertainty responses get more samples, low-uncertainty get fewer. Achieves 50% sample efficiency improvement and 12.6% AUROC gain over uniform sampling baseline.
- **Relevance to Carnot:** NUP Probe v1 (Exp 484) got AUC=0.600 using character-entropy fallback (no logprobs). Bayesian adaptive entropy estimation could be integrated as richer features: instead of fixed-threshold character entropy, use Bayesian uncertainty bands to estimate whether continuation entropy is "confidently high" or "uncertainly medium." This should push NUP Probe beyond the 0.700 Tier 0c threshold.
- **Concrete experiment:** NUP Probe v2 (RETRO-047): Replace character-entropy proxy with Bayesian semantic entropy estimator. Evaluate AUC on accumulated real CoT pairs from Exps 488/489.
- **When to incorporate:** Milestone 2026.04.37 — NUP Probe enrichment (Exp 496).

### SuRe — Surprise-Driven Prioritized Replay for Continual EBM Learning
- **Paper:** arXiv 2511.22367 (2025)
- **What:** Surprise-prioritized replay for continual learning: selects high-NLL (negative log-likelihood) sequences for replay, meaning sequences the model was most "surprised" by. Combines fast and slow LoRA adapters with exponential moving average merging, achieving +5% accuracy on continual learning benchmarks. The surprise metric naturally identifies domain boundary violations.
- **Relevance to Carnot:** PPSEBM (Exps 470, 485) uses domain-partitioned EBM learning to prevent catastrophic forgetting. SuRe's surprise metric could replace the current fixed-priority scheme for selecting which constraint violations to replay: arithmetic violations that surprised the model (high energy, low prior probability) get prioritized for replay, reducing forgetting at domain boundaries. This is a Tier 2 self-learning improvement.
- **Concrete experiment:** Add SuRePriorityReplay to PPSConstraintLearner: rank constraint violations by energy surprise (EBM energy - expected energy from prior session), replay top-k per domain partition. Compare isolation_score before/after vs uniform replay baseline.
- **When to incorporate:** Milestone 2026.04.37 — Tier 2 self-learning experiment (Exp 495).

## 2026-04-19 arxiv Scan (Milestone 2026.04.36 Planning)

### Neural Uncertainty Principle — Hallucination as Under-Constrained Continuation
- **Paper:** arXiv 2603.19562 (2026)
- **What:** Frames hallucination as an under-constrained continuation problem. When a model is about to hallucinate, the entropy of its next-token distribution is high because multiple continuations are nearly equally likely. Correct continuations are highly constrained by logical and factual dependencies.
- **Relevance to Carnot:** Directly compatible with Carnot's energy-based formulation. High continuation entropy = high energy = violation likelihood. Enables NUPProbe: a zero-latency Tier 0c pre-filter that requires no LLM call — just entropy computation on logprobs. AUC > 0.700 qualifies it for the verification cascade.
- **Incorporated as:** Exp 484 (Milestone 2026.04.36)

### Verifying Chain-of-Thought via Computational Graph (CRV)
- **Paper:** arXiv 2510.09312 (2025)
- **What:** Circuit-based Reasoning Verification proposes a computational graph formalism for LLM CoT, detecting structural failures (wrong intermediate computations, invalid deductions) via graph reachability and cycle detection.
- **Relevance to Carnot:** Complementary to VeriCoTStepValidator (Exp 453). The graph-based view could extend VeriCoT from per-step arithmetic checking to multi-step logical dependency verification — the domain where the global consistency checker (Exp 172) showed 100% detection.
- **When to incorporate:** When extending VeriCoT to multi-step dependency checking (Phase 3 pipeline work).

### Typed Chain-of-Thought — Formal Certification for LLM Reasoning
- **Paper:** arXiv 2510.01069 (2025)
- **What:** Assigns formal types to CoT reasoning steps (arithmetic, logical, factual) and certifies transitions between types. Incompatible type transitions flag reasoning failures.
- **Relevance to Carnot:** The PPSEBM domain classification (arithmetic/code/logical) is a simplified version of this approach. Typed CoT's formal certification framework could improve PPSEBM's domain boundary detection, reducing false positives on cross-domain steps.
- **When to incorporate:** When PPSEBM real-data validation (Exp 485) reveals specific domain boundary failure modes.

### GSM-Symbolic — Adversarial Robustness Benchmark
- **Paper:** arXiv 2410.05229 (ICLR 2025, Apple)
- **What:** Generates benchmark instances from symbolic templates with identical logical structure but different surface forms (numbers, irrelevant context sentences). ALL tested LLMs drop significantly: o1-preview 92.7%→77.4%, GPT-4o 95%→88%, Llama3-70B 90%→75%.
- **Relevance to Carnot:** The thesis experiment: Carnot's improvement should be LARGER on adversarial variants because Ising verifies arithmetic constraints independently of irrelevant context. This is the headline credibility result.
- **Incorporated as:** Exp 479 (Milestone 2026.04.36)

### LSEBMCL — EBM Replay for Continual Constraint Learning
- **Paper:** arXiv 2501.05495 (2025)
- **What:** Learning Symmetric Energy-Based Models with Continual Learning. EBM replay (warm-starting from prior session's coupling matrix) prevents catastrophic forgetting when constraint distributions shift across sessions.
- **Relevance to Carnot:** Foundation for Tier 2 self-learning (ConstraintAdditionFromMemory). Exp 457 implemented LSEBMCL baseline achieving session2_fp_rate=0.0.
- **Incorporated as:** PPSEBM baseline (Exps 470, 485)

### PPSEBM — Progressive Parameter Selection EBM
- **Paper:** arXiv 2512.15658 (2025)
- **What:** Progressive Parameter Selection adds domain-partitioned weight updates to EBM continual learning. Each constraint domain (arithmetic/code/logical) gets an isolated parameter partition; updates to one domain don't modify others.
- **Relevance to Carnot:** Directly implements Tier 2 self-learning isolation. Prevents the case where improving arithmetic detection accidentally degrades code detection (catastrophic interference across constraint domains).
- **Incorporated as:** Exps 470 (milestone .35), 485 (milestone .36 real-data validation)

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

### PPSEBM — EBM with Progressive Parameter Selection for Continual Learning
- **Paper:** arxiv.org/abs/2512.15658 (2025-12)
- **What:** Combines Progressive Parameter Selection (PPS) with EBM-based generative replay to
  prevent catastrophic forgetting in continual LLM fine-tuning. Each new task gets task-specific
  parameters isolated from previous tasks; the EBM generates pseudo-samples from prior tasks to
  guide the parameter selection. Uses Mistral 7B as base LLM. Outperforms state-of-the-art
  continual learning methods on NLP benchmarks.
- **Relevance:** Directly extends LSEBMCL (Exp 457, arXiv 2501.05495). Where LSEBMCL uses EBM
  replay for warm-starting constraint templates, PPSEBM adds task-specific parameter isolation —
  each constraint domain (arithmetic, code, logic) gets its own parameter partition. This is
  Tier 2 self-learning done right: learned parameters from "arithmetic session" do not interfere
  with learned parameters from "code session." The EBM generates synthetic prior-session violations
  to reinforce parameter boundaries — directly applicable to Carnot's cross-session constraint memory.
- **When to pursue:** Next milestone. Implement PPSConstraintLearner as an upgrade over the
  LSEBMCL replayer: add task-specific parameter partitions for each constraint domain, use the
  EBM to generate boundary violations for reinforcement.

### Equilibrium Propagation on Oscillator Ising Machines
- **Paper:** arxiv.org/abs/2510.12934 (2025-10)
- **What:** Equilibrium Propagation (EP) is a local backprop alternative suited to physical systems
  that naturally perform energy descent. Demonstrates EP training on Oscillator Ising Machines (OIMs)
  achieving ~97.2% MNIST accuracy. OIMs are GHz-frequency physical systems where energy descent
  mirrors gradient descent on loss landscapes. Robust to parameter quantization and phase noise.
- **Relevance:** Carnot's Ising sampler (Exp 46, 61) performs energy descent — it IS an Ising
  machine. EP training on OIMs means Carnot's Ising EBM could learn coupling updates locally
  (no backprop) via the same physics that drives sampling. This is the Tier 1 online weight
  update done at sampling speed. The OIM's natural energy descent IS the learning rule.
  Also directly relevant to FPGA bring-up: FPGA Ising machines are digital OIM simulations.
- **When to pursue:** FPGA hardware milestone. Add EP-style local coupling update rule to
  IsingEBM (update couplings based on difference between free-phase and clamped-phase spin
  correlations — no gradient computation needed).

### How to Train an OIM using Equilibrium Propagation
- **Paper:** arxiv.org/abs/2505.02103 (2025-05)
- **What:** Practical guide to OIM training via EP on CMOS hardware. Achieves competitive accuracy
  with 10-bit parameter precision and 4-bit phase detection. Moderate phase noise can enhance
  performance (stochastic resonance). Demonstrates feasibility for physical OIM implementations.
- **Relevance:** Companion to arXiv 2510.12934. Provides the quantization parameters for FPGA
  implementation of EP-trained Ising machines. The 10-bit / 4-bit precision targets are directly
  implementable in LUT-based FPGA Ising machines (like the KV260 design).
- **When to pursue:** KV260 FPGA bitfile design. Use 10-bit coupling precision and 4-bit phase
  measurement as the implementation target for the sparsified Ising sampler bitfile.

### GPU-Accelerated Simulated Oscillator Ising/Potts Machine
- **Paper:** arxiv.org/abs/2505.22631 (2025-05)
- **What:** GPU simulation of Oscillator Ising/Potts machines achieving ~10,000x speedup over
  CPU heuristics on combinatorial optimization. 1024-neuron all-to-all connected probabilistic
  Ising accelerator. Demonstrates that GPU simulation of Ising dynamics is a practical path to
  fast constraint sampling without custom hardware.
- **Relevance:** Carnot has 2x RTX 3090 (48GB VRAM total) currently underutilized (GPU 1 idle
  throughout milestone .34). GPU-accelerated Ising simulation would replace CPU-based
  ParallelIsingSampler with a 10,000x faster GPU variant — directly on existing hardware.
  This unblocks Tier 3 JEPA predictor (which needs fast energy evaluation for real-time gating)
  without waiting for FPGA or TSU hardware.
- **When to pursue:** Next milestone. Implement GPUOscillatorIsingSimulator on RTX 3090 and
  compare vs ParallelIsingSampler (CPU). Expect significant speedup for n_vars > 100.

### From Ising to Potts: Physics-Inspired Potts Machines for Low-Energy Sampling
- **Paper:** arxiv.org/abs/2507.18379 (2025-07)
- **What:** Extends Ising machine (binary spins) to Potts machine (multi-state spins, k states).
  Coupled oscillator implementation achieves low-energy sampling for combinatorial optimization.
  Multi-state spins can encode more complex constraint types than binary — e.g., arithmetic
  carry (3 states: no-carry, carry-generate, carry-propagate) without binary expansion.
- **Relevance:** Carnot's IsingEBM uses binary spins (σ ∈ {-1, +1}). Many constraint types
  are naturally multi-valued: numeric ranges, categorical types, enumerated error classes.
  A PottsEBM subclass with k ∈ {3, 4, 8} states would allow richer constraint encoding without
  expanding the variable count (which MCMC mixing time increases with). Pairs naturally with
  MFC sparse coupling (arXiv 2602.04200) for FPGA-efficient multi-state constraints.
- **When to pursue:** Energy tier expansion milestone. Implement PottsEBM as a new model tier
  between Ising (binary) and KAN (continuous). Benchmark on constraint types that benefit from
  multi-state encoding (arithmetic carry, unit type consistency, range bounds).

### GSM-Symbolic — Apple's Adversarial Math Reasoning Benchmark (THE CREDIBILITY TEST)
- **Paper:** arxiv.org/abs/2410.05229 (2024-10, ICLR 2025)
- **What:** Apple researchers created GSM-Symbolic by generating new instances from symbolic
  templates (same logical structure, different numbers and irrelevant sentences). ALL tested
  LLMs showed significant accuracy drops. o1-preview dropped from 92.7% → 77.4%. 8-shot
  prompting didn't help. Demonstrates that LLMs pattern-match, not reason.
- **Relevance:** THIS IS CARNOT'S THESIS EXPERIMENT. Carnot's verify-repair loop should
  maintain accuracy on GSM-Symbolic because it verifies arithmetic via external Ising constraints,
  not by prompting. The expected result: (1) LLM baseline drops on symbolic variants, (2)
  Carnot's verify-repair closes the gap because Ising catches arithmetic errors regardless of
  irrelevant context. This is the headline benchmark Carnot needs for credibility.
- **Status:** Confirmed exists (ICLR 2025). Run against the adversarial variant GSM-Symbolic,
  not just GSM8K. Download via: datasets.load_dataset('apple/GSM-Symbolic', 'main') or similar.
- **When to pursue:** NEXT MILESTONE — HIGHEST PRIORITY credibility experiment.

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

### Carnot for Code-Vulnerability Verification — Validation Moat Spike (STRETCH, 2026-04-19)
- **Directive:** As a post-milestone-2026.04.40 stretch direction, apply
  Carnot's verify-repair pipeline to code-vulnerability discovery and compare
  against the Vidoc Security replication study
  (https://decrypt.co/364744/anthropic-mythos-replicated-public-models-vidoc-security,
  April 2026). Vidoc showed public models (GPT-5.4, Claude Opus 4.6) find
  candidate vulnerabilities for under $30/scan but "cannot construct working
  attack chains — they lack the sophistication to figure out how an attacker
  could chain code fragments together across multiple network packets". That
  chain-the-fragments gap is exactly what Carnot's Ising+Z3 symbolic layer
  closes.
- **Proposed experiment:** `CarnotVulnVerifier` — takes Vidoc's public replay
  of the OpenBSD, FFmpeg, and wolfSSL test cases as input (candidate bug
  surfaces from a cheap LLM detector), compiles each candidate into a Z3
  constraint against the surrounding call graph, runs the Ising solver to
  check whether the joint constraint is satisfiable (= real bug chain) or
  unsatisfiable (= false positive). Emit per-case `verdict in {real_chain,
  false_positive, undetermined}` plus the Z3 witness when real_chain.
  Acceptance criterion: at least one of the OpenBSD/FFmpeg/wolfSSL cases
  converts a public-model false positive to a certified false_positive, or
  lifts a partial discovery to a verified real_chain with an exploit
  trace. Report cost_usd and decision_class=verify honestly per the new
  economics fields in `scripts/experiment_template.py`.
- **Why this matters beyond the specific experiment:** it is the first test
  of Carnot outside the LLM-hallucination domain. The symbolic layer is
  domain-agnostic by design; proving that on a second, adjacent domain
  (code vulns) lets the README / landing page make a stronger claim about
  the validation-moat framing. It also gives a headline metric that does
  not depend on GSM8K or HumanEval, which matters because RETRO-033's
  ongoing miss count has made the LLM-math headlines currently unlandable.
- **Not a primary focus:** Carnot's core is LLM verification; do not let
  the security vertical become the headline. The point of this spike is to
  prove the verification layer is AI-model-independent. One milestone slot
  max, scheduled only after the pyxrt NPU work (Exp N+3/N+4 in the
  env-hardening proposal) has landed.
- **Watch-for signals** that would upgrade this from stretch to priority:
  AMD or another vendor publishes a VitisAI Linux wheel (currently blocked
  per the earlier env-hardening reference); a second independent study
  replicates Vidoc's $30/scan number on a different vulnerability class;
  the Z3 path in Exp 453 produces a clean extension interface that a
  non-math domain could plug into without rewriting the extractor.

### Environment Hardening + Stranded Experiment Reruns (CRITICAL, 2026-04-19)
- **Directive:** Milestone 2026.04.38 lost headline-credibility experiments
  to environment gaps the conductor has no pre-flight check for. Exp 503
  blocked on concurrent-process VRAM OOM (a new failure mode, not a zombie —
  an unrelated Python process joined between the GPUVRAMGateV2 check and
  the model load and won the VRAM race). Exp 504 blocked on three missing
  Python packages (sentencepiece, tiktoken, llama-cpp-python) that were
  imported without being in any dependency manifest.
- **Manual unblock completed 2026-04-19:** six stale pytest processes
  killed (~48 GiB VRAM freed, both GPUs now 24,123 MiB free), three
  missing deps installed into the project venv. Exps 503 and 504
  remain marked `blocked` in their deliverable JSON and need a proper
  rerun under the fixed env.
- **Three experiments proposed for milestone 2026.04.39** (full design in
  `openspec/change-proposals/env-hardening-and-reruns.md`):
  1. Conductor startup env check — verify imports, GPU VRAM headroom,
     no stale pytest workers, disk space, git remote reachable. Catches
     the exact 503/504 failure modes before the subagent is invoked.
  2. Zombie pytest reaper + rerun of Exps 503 and 504 under the fixed
     env; report `success` or `blocked_again` with specific reason.
  3. **JEPA Live Retrain v4 rerun (hard-sequenced after #2).** The
     stranded Exp 510 is not a code failure — the committed script is
     already correct — it is a data failure: `n_live_pairs=0` because
     its upstream (503/504) was blocked. Rerunning is the validation
     gate for the 0.967 AUC headline claim in `docs/roadmap.md`. Must
     run after #2 succeeds; a premature rerun just re-produces the
     synthetic-only result. Acceptance criterion: deliverable has
     `n_live_pairs > 0` AND `honest_verdict` is one of
     `fr11_live_confirmed / fr11_live_regressed / fr11_live_insufficient_signal`.
  4. **NPU `backend="pyxrt"` rewrite of Exp 511.** A web audit on
     2026-04-19 confirmed that the Linux ONNX+VitisAI path is
     structurally blocked and unsupported by AMD. Ryzen AI Software
     1.7.1 is Windows-only; the `voe` wheel is missing on Linux;
     source-building onnxruntime with `--use_vitisai` fails on
     GCC 13/14; HuggingFace Optimum-AMD depends on the same missing
     pieces. See GitHub issues
     `amd/RyzenAI-SW#341`, `amd/xdna-driver#1017`,
     `microsoft/onnxruntime#27097`, `amd/RyzenAI-SW#319` — all open,
     all unanswered by AMD. The realistic Linux NPU path is
     `pyxrt` direct (verified working 2026-04-19,
     `results/npu_stack_verification.json` =
     `npu_fully_available_pyxrt_only`). Rewrite Exp 511's
     `NPUEntropyProbe` with a `backend="pyxrt"` mode and rerun.
  5. **Exp 460 extension: mlir-aie / IRON backend.** In parallel with
     #4, install the `mlir_aie` + `llvm-aie` pip packages (v1.3.1,
     March 2026, AMD-maintained) and port one of the repo's softmax
     example kernels to run through `pyxrt`. Adds a second
     Linux-native NPU backend so the project is not single-path on
     pyxrt alone.
  6. (Stretch) Dependency lockfile + `scripts/bootstrap.sh` via AST
     walk of imports in `scripts/` and `python/`, so "missing package"
     becomes a two-line diagnostic rather than a silent mid-experiment
     block.
- **Why:** the whack-a-mole pattern has shifted from "zombie VRAM"
  (closed by GPUVRAMGateV2) to "concurrent-process contention + missing
  deps". Without these new pre-flight checks, RETRO-033/038/039 will
  continue to miss, not on research quality but on environment hygiene.

### Conductor Regression Prevention — Harness Hardening (HIGH PRIORITY, 2026-04-18)
- **Directive:** Before scheduling more research-heavy milestones, the planner should
  schedule four infrastructure experiments that harden `scripts/research_conductor.py`
  itself. Milestone 2026.04.33 lost ~7.5 hours to a silent fail-counter bug and
  another ~3 hours to wall-clock kills of subagents that had already produced their
  deliverables. See the full design doc at
  `openspec/change-proposals/conductor-regression-prevention.md`.
- **The four experiments (name them Exp N..N+3 sequentially):**
  1. Regression tests for `pick_next_task` (title-prefix whitespace boundary) and
     `run_agent` (deliverable-watch kill-switch). Pins both bugs with unit tests.
  2. Startup invariant check — at conductor boot, round-trip every task title
     through `log_step` → `pick_next_task` key matching; `sys.exit(2)` with a clear
     diagnostic if any task fails the round-trip. Turns a silent infinite loop into
     a boot-time error.
  3. Structured event log — emit `ops/conductor-events.jsonl` per state transition
     plus a `scripts/conductor_health.py` CLI that summarises cadence, kill-reason
     distribution, and flags any task appearing more than MAX_FAILURES_PER_TASK times.
  4. Fake-Claude state-machine test harness — `FakeAgentBackend` injectable into
     `run_agent`, driven by a scripted roadmap fixture that exercises every failure
     path (timeout, max-turns, half-written deliverable, boundary title, etc.) in
     milliseconds.
- **Why:** The research code itself was never the bottleneck in the last milestone.
  The orchestration harness was. Without this hardening the next long-autonomous run
  will hit a similar class of bug and burn another multi-hour window before the user
  notices.
- **Rollout:** Experiments 1+2 land together in milestone 2026.04.35 (tiny, bundle
  them). Experiment 3 lands in 2026.04.36. Experiment 4 in 2026.04.37 once there is
  enough event-log data to write the assertions against. All four are CPU-only and
  should NOT displace the headline verify-repair benchmarks in any given milestone.

### SOTA Local GGUF Models — Mandated Model Set (HIGHEST PRIORITY, 2026-04-18)
- **Directive:** All new experiments that exercise an LLM (verify-repair, live benchmarks,
  CoT, adversarial, precision, HumanEval, GSM8K, FOVER, etc.) MUST include at least one of
  the following frontier local models in their `MODEL_SPECS` list. Legacy small models
  (Qwen3.5-0.8B, Gemma4-E4B-it) are no longer acceptable as headline results — they may
  only be used for fast CPU smoke-tests or cheap reproduction runs.
- **Mandated SOTA GGUF models** (hosted on HuggingFace by unsloth, quantized):
  - `unsloth/Qwen3.6-35B-A3B-GGUF` — Qwen 3.6 35B MoE, ~3B active params per token.
    Flagship MoE choice; highest capability per compute for reasoning / verify-repair.
  - `unsloth/gemma-4-26B-A4B-it-GGUF` — Gemma 4 26B MoE, ~4B active params, instruction-tuned.
    Middle-tier MoE choice; strong instruction-following.
  - `unsloth/gemma-4-31B-it-GGUF` — Gemma 4 31B dense, instruction-tuned. Flagship dense
    choice; use when MoE routing overhead hurts latency or when dense activation is needed
    (e.g. full attention for adversarial probes).
- **Why:** These are the most capable local models available as of April 2026. Carnot's
  headline verify-repair numbers need to be on the frontier to be taken seriously; Qwen3.5-0.8B
  and Gemma-4-E4B-it are only appropriate for smoke-tests going forward.
- **Loading path:** GGUF → llama.cpp backend (the same path being stabilized by Exp 450's
  Gemma4 tokenizer fix). When Exp 450 lands, all three should load cleanly via
  `llama-cpp-python`. Until then, the two Gemma-4 GGUFs may still hit the infinite-<unused>
  bug; validate tokenizer output on load or fall back to the HF transformers loader.
- **Rollout plan:**
  1. Exp 450 closes the Gemma4 tokenizer bug (in-flight, milestone 2026.04.34).
  2. Exp 451 re-runs the live precision benchmark using one of these SOTA models as the
     primary judge + one as the verifier. This becomes the first credible live
     verify-repair number.
  3. Downstream experiments in 2026.04.34+ adopt the same model set.
- **Hardware implications:** A3B MoE runs comfortably on a single 24 GB GPU at Q4_K_M; the
  31B dense needs Q4_K_M on 24 GB or Q5_K_M on 32 GB; the 26B A4B MoE fits in ~16 GB at
  Q4_K_M. Plan DualGPURunner scheduling so one 24 GB GPU runs the large model and the other
  runs the smaller verifier concurrently.

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

## 2026-04-22 arxiv Scan (Milestone 2026.04.54 Planning)

### SC-Energy — Set Consistency Energy Networks (arXiv 2503.10695)
- **Paper:** arXiv 2503.10695 (March 2025)
- **What:** Introduces SC-Energy, a neural energy model that assesses logical coherence
  of entire statement sets using contrastive learning. Unlike pairwise verification, SC-Energy
  computes a global consistency score over N statements simultaneously, enabling detection of
  contradictions that span multiple reasoning steps.
- **Relevance to Carnot:** Extends SymCodeVerifier (Tier 2.5) and CausalReasoningVerifier
  (Tier 2.7) which check pairwise step transitions. SC-Energy could catch "correct arithmetic
  at each step, but globally contradictory conclusion" errors — the class of errors that
  neither tier currently detects. Tier 2.9 candidate.
- **Concrete experiment:** Implement SetConsistencyVerifier wrapping SC-Energy on GSM8K CoT
  step sets. Compare AUC on FoVer formal v1 corpus vs SymCodeVerifier and CausalVerifier.
  Target: AUC >= 0.75 on multi-step contradiction detection.
- **When to incorporate:** Milestone 2026.04.54 — Phase 4 new research (Exp 711).

### PRM Planning Dataset — PDDL-Based Step-Level Reward Synthesis (arXiv 2604.17957)
- **Paper:** arXiv 2604.17957 (April 2026)
- **What:** Uses PDDL formal planning to automatically generate ~1M step-level reward labels
  for PRM training. A step is labeled correct iff it is a valid plan transition. Achieves
  cross-domain generalization (math, planning, NLI) without human annotation. Demonstrated
  on GSM8K, MATH, and PDDL-structured tasks.
- **Relevance to Carnot:** Carnot's JEPA predictor is bottlenecked by FoVer formal v1
  (200 Z3-labeled pairs). PDDL-based synthesis could scale the training corpus 5-10x without
  human effort, addressing the persistent data bottleneck. Z3 (Exp 686) and PDDL are
  complementary — Z3 handles arithmetic entailment, PDDL handles procedural planning steps.
- **Concrete experiment:** Implement PDDL step synthesizer for GSM8K word problems (encode
  state as noun-quantity pairs, transitions as arithmetic operations). Generate 1000 labeled
  pairs. Combine with FoVer Z3 labels → fover_v2_combined.json (1200+ pairs total).
  Use as JEPA v17 training data.
- **When to incorporate:** Milestone 2026.04.54 — Phase 4 new research (Exp 712).

### Multi-Domain RL for Cross-Model Verification Generalization (arXiv 2602.12566)
- **Paper:** arXiv 2602.12566 (February 2026)
- **What:** Investigates how verification/reward models trained on one domain (math, code,
  science) transfer to others under reinforcement learning. Key finding: models trained on
  mixed domains show better cross-model generalization than single-domain training, and
  "synergistic mixing" (math + code + science together) outperforms any single-domain
  specialist.
- **Relevance to Carnot:** Carnot's VR pipeline works for Qwen3.5-0.8B (signed_improvement=1.0)
  but HURTS Gemma4-E4B-it (signed_improvement=-0.8). The cross_model_delta=-1.8 is a
  critical failure. This paper explains why: the SymCodeVerifier and constraint extractors
  were calibrated on Qwen-format arithmetic chains, making them incompatible with Gemma's
  reasoning format. The fix: multi-domain training with Gemma-format chains as a second
  training domain, or model-specific threshold adaptation.
- **Concrete experiment:** Diagnose Gemma4 VR failure by tracing which pipeline step
  causes harm (Exp 706). Then implement model-adaptive constraint thresholds: suppress
  constraint types with FP rate > TP rate for Gemma specifically. Target: Gemma4
  signed_improvement >= 0 (remove harm while preserving Qwen win).
- **When to incorporate:** Milestone 2026.04.54 — Phase 2 Gemma VR fix (Exps 706-708).

## 2026-04-22 arxiv Scan (Milestone 2026.04.57 Planning)

### CoCoA — Inter-Layer Disagreement for Hallucination Mitigation (arXiv 2602.09486)
- **Paper:** arXiv 2602.09486 (February-March 2026)
- **What:** CoCoA (Confusion and Consistency Aware) decoder is a training-free decoding
  algorithm that mitigates hallucinations by measuring representational instability across
  the model's middle hidden layers. At each decoding step, candidate spans are scored by
  their ConMLDS (Contrastive Multi-Layer Disagreement Score) — how much the hidden state
  representation shifts between early and late layers. High disagreement = likely hallucination.
  Works across Llama-3, Qwen-2.5, Mistral. Tasks: QA, summarization, math, code.
- **Relevance to Carnot:** Orthogonal to all existing Tier 0 probes. The Tier 0 stack
  (NUP, HalluField, BasinDetector) operates on energy/logit space. CoCoA operates on
  representational geometry between layers — a fundamentally different signal source.
  Can be computed from the same forward pass as JEPAReasonerProbe (shares layer activations).
  Zero training required — immediate deployment.
- **Concrete experiment (Exp 743):** Implement CoCoADetector using Qwen3.5-0.8B layers
  8-16 (middle third). Compute ConMLDS per query. Evaluate AUC on FoVer v2.
  Wire as Tier 0f (advisory, after HalluField, before SinkProbe). Compare AUC to
  existing Tier 0 probes. Hardware path: pure CPU matrix ops — FPGA-compatible.
- **When to incorporate:** Milestone 2026.04.57 — Phase 2 new capabilities.

### Fully Parallel Probabilistic Ising Machine on FPGA (arXiv 2604.17109)
- **Paper:** arXiv 2604.17109 (April 2026)
- **What:** Hardware-software co-design of a probabilistic Ising machine on FPGA using
  Vitis High-Level Synthesis (HLS). C++ kernel with loop pipelining, loop unrolling, and
  memory partitioning achieves fully parallel hardware implementation. Key: uses Vitis HLS
  (separate from full Vivado) to generate RTL from C++ — opens a synthesis path that
  doesn't require installing the full Vivado IDE.
- **Relevance to Carnot:** KV260 synthesis has been blocked for 3 consecutive milestones
  (Exps 584, 701, 701) because Vivado is not installed. Vitis HLS is distributed separately
  (available via AMD Vitis installer or as part of AMD's Docker images). If Vitis HLS can
  be installed, this paper's C++ HLS approach could synthesize the Ising sampler without
  needing the full Vivado GUI. Also validates the parallel checkerboard architecture
  already in ising_sampler_v2.v.
- **Concrete experiment (Exp 750):** Write ising_sampler_hls.cpp based on arXiv 2604.17109
  C++ kernel pattern. Install Vitis HLS (check `vitis_hls --version`). If available:
  synthesize. If not: write the C++ kernel as simulation that matches the RTL spec.
  Either way: validate the HLS approach matches Python simulation results.
- **When to incorporate:** Milestone 2026.04.57 — Phase 4 hardware frontier.

### Iterative Self-Repair in Code Generation (arXiv 2604.10508)
- **Paper:** arXiv 2604.10508 (April 2026)
- **What:** Investigates iterative self-repair across 7 LLMs on HumanEval and MBPP.
  Key findings: (1) self-repair universally improves pass rates by +4.9 to +17.1pp on
  HumanEval and +16.0 to +30.0pp on MBPP; (2) most gains concentrate in the first 2
  rounds; (3) assertion errors are hardest (~45% repair rate), syntax/name errors easiest.
  The gains are model-size independent — even small models benefit from 2-round repair.
- **Relevance to Carnot:** Carnot's code verification uses execution-based checking
  (CodeExtractor + runtime instrumentation), which is the right approach. But the repair
  step is currently single-round. This paper proves 2-round repair captures ~90% of
  the total available improvement. Implementing 2-round repair in the VerifyRepairPipeline
  for code tasks could yield +4.9-17.1pp HumanEval improvement — the largest single
  improvement potential currently identified in the literature for our architecture.
- **Concrete experiment (Exp 744):** Implement TwoRoundCodeRepairPipeline:
  round1=(generate, execute, repair if fail), round2=(re-execute, repair if still fail).
  Benchmark on HumanEval with Qwen3.5-0.8B (baseline: last HumanEval run).
  Measure per-round improvement. Compare error type distribution (syntax vs assertion).
- **When to incorporate:** Milestone 2026.04.57 — Phase 2 new capabilities.

### SETS: Self-Verification and Self-Correction for Test-Time Scaling (arXiv 2501.19306)
- **Paper:** arXiv 2501.19306 (January 2026)
- **What:** Self-Enhanced Test-Time Scaling combines parallel and sequential techniques.
  LLM first generates multiple candidate solutions in parallel, then applies self-verification
  to select the best candidate, then applies self-correction if the selected candidate
  has issues. Outperforms both pure parallel (repeated sampling) and pure sequential
  (SELF-REFINE) approaches. Works for reasoning and code tasks.
- **Relevance to Carnot:** Carnot's verify-repair pipeline is a form of sequential
  test-time scaling. SETS suggests combining parallel sampling (generate N candidates)
  with Carnot's constraint-based verification (select lowest-energy candidate) then
  targeted repair. This is the energy-guided best-of-N architecture: generate 4-8
  candidates, score by cascade energy, repair the lowest-energy one if still failing.
  This should significantly improve VR improvement over single-candidate pipelines.
- **Concrete experiment:** Energy-guided best-of-N + repair. Generate 4 candidates,
  score by Tier 2.1 probe energy, select lowest, repair if needed. Compare to single
  candidate baseline. (Future milestone — after 2-round repair is validated.)
- **When to incorporate:** Milestone 2026.04.58 (after 2-round repair is proven in .57).

### D-Wave Simulated Annealing (dwave-ocean-sdk neal) as SamplerBackend
- **Tool:** dwave-ocean-sdk (Apache 2.0), pip-installable
- **What:** D-Wave's open-source SDK includes `neal.SimulatedAnnealingSampler` — a
  CPU-based simulated annealing solver for QUBO/Ising problems. Runs entirely locally,
  no QPU access required. Same API as D-Wave QPU backends: submit BinaryQuadraticModel,
  get SampleSet back. Also includes `dimod` for problem formulation and `greedy` for
  steepest descent.
- **Relevance to Carnot:** The SamplerBackend abstraction (Exp 71) was designed for
  pluggable backends (CPU, TSU, future QPU). D-Wave's neal provides a third backend
  with a fundamentally different algorithm (simulated annealing vs Gibbs sampling).
  For dense constraint graphs where Gibbs gets stuck in local minima, SA may find
  lower-energy solutions. Comparing neal vs ParallelIsingSampler on real constraint
  problems from GSM8K violations validates the backend abstraction.
- **Concrete experiment (Exp 751):** pip install dwave-ocean-sdk. Implement
  DWaveNealBackend(SamplerBackend). Test on 20 constraint problems from GSM8K violations
  (convert IsingEBM couplings to BQM format). Compare solution quality (final energy)
  and speed vs ParallelIsingSampler. This is $0 cost and unlocks quantum backend path.
- **When to incorporate:** Milestone 2026.04.57 — Phase 4 hardware frontier.

## 2026-04-23 arxiv Scan (Milestone 2026.04.59 Planning)

### EBRM — Energy-Based Reward Models for Robust LLM Alignment (arXiv 2504.13134) ⭐ CRITICAL
- **Paper:** arXiv 2504.13134 (April 2025)
- **What:** EBRM learns an energy function over rewards + embeddings, modeling the full
  noisy signal distribution rather than a point estimate. Outperforms discriminative reward
  models on chat-hard, safety, and reasoning benchmarks.
- **Relevance to Carnot:** FOUNDATIONAL VALIDATION. Carnot's EORM is an energy-based reward
  model for step-level verification. This paper is the closest prior work — it validates
  the EORM architecture theoretically. Cite in all Carnot publications. Direct comparison:
  does Carnot's step-level EORM outperform EBRM on verification tasks?
- **Concrete experiment:** Implement EBRM baseline on FoVer labeled steps. Compare AUC
  to EORM. If EORM outperforms on step-level tasks (different from EBRM's token-level
  reward), this is publishable.
- **When to incorporate:** Milestone 2026.04.59 — Phase 3 new research (Exp 771).

### SETS — Self-Verification + Self-Correction for Test-Time Scaling (arXiv 2501.19306) ⭐
- **Paper:** arXiv 2501.19306 (January 2026, updated December 2025)
- **What:** SETS (Self-Enhanced Test-Time Scaling) unifies parallel BoN sampling with
  sequential self-correction. The LLM first generates N candidates, uses zero-shot
  self-verification to select the best, then applies self-correction if needed.
  Outperforms pure BoN and pure self-refine without external verifier.
- **Relevance to Carnot:** Carnot's energy-guided BoN + repair is the hardware-accelerated
  version of SETS, with energy scores replacing zero-shot self-verification. Head-to-head
  comparison on HumanEval/GSM8K will quantify Carnot's advantage over the SETS baseline.
  If Carnot wins on efficiency (energy/token), this is a core publishable result.
- **Concrete experiment:** Run SETS and Carnot on identical 50-problem HumanEval split.
  Compare: pass rate, oracle calls, total inference time. Hypothesis: Carnot uses fewer
  oracle calls to reach same or better pass rate (energy is cheaper than LLM self-verify).
- **When to incorporate:** Milestone 2026.04.59 — Phase 3 new research (Exp 773).

### Semantic Energy — EBM Hallucination Detection Beyond Entropy (arXiv 2508.14496) ⭐
- **Paper:** arXiv 2508.14496 (August 2025, updated December 2025)
- **What:** Logits carry stronger uncertainty signal than probabilities because they
  retain intensity information lost during softmax normalization. Semantic energy
  E = -log p(x) evaluated over semantic equivalence classes outperforms standard
  semantic entropy for hallucination detection. Works on TriviaQA, NQ, SciQ.
- **Relevance to Carnot:** Carnot's Tier 0 stack uses logit-based energy probes. Semantic
  energy is the theoretically grounded formulation of what Carnot is already doing.
  Implementing the semantic energy formula as a Tier 0g probe could improve AUC over
  the current logprob heuristics and would provide theoretical grounding for publications.
- **Concrete experiment:** Implement SemanticEnergyProbe using logit-space energy formula.
  Evaluate AUC on FoVer v2 vs current NUP probe and SpilledEnergyDetector.
  Target: AUC >= 0.97 (matching HalluField's synthetic result).
- **When to incorporate:** Milestone 2026.04.59 — Phase 3 new research (Exp 772).

### Adaptive Bayesian Hallucination Detection (arXiv 2603.22812)
- **Paper:** arXiv 2603.22812 (March 2026)
- **What:** Adaptive Bayesian framework for semantic entropy. Dynamically adjusts sampling
  count using variance-based thresholds — stops early when variance drops below criterion.
  Achieves 50% fewer samples at comparable detection quality vs fixed-budget sampling.
- **Relevance to Carnot:** PSV sampling loop is currently fixed-budget (K parallel samples).
  Adaptive variance-based stopping could reduce oracle calls in PSV by 30-50% without
  hurting repair quality. Directly lowers the compute cost of the verify-repair pipeline.
- **Concrete experiment:** Add AdaptiveSamplerConfig(variance_threshold=0.05) to PSV.
  Measure: mean samples per question vs current fixed-K=4. Measure detection AUC at each
  K level. Report: sample reduction fraction and AUC delta at threshold.
- **When to incorporate:** Milestone 2026.04.59 — Phase 3 new research (Exp 774).

### Jailbreak Detection via Latent Internal Representations (arXiv 2602.11495)
- **Paper:** arXiv 2602.11495 (February 2026)
- **What:** Lightweight linear classifiers trained on hidden state features from specific
  transformer layers reliably distinguish jailbreak prompts from benign ones. Requires no
  retraining of the base model. Works across Llama-3, Qwen-2.5, Mistral. Identifies
  which layers carry the adversarial signal.
- **Relevance to Carnot:** Carnot's EORM already analyzes hidden states at multiple layers.
  Adding a jailbreak head to the EORM feature extractor costs nothing extra (shared
  forward pass). Product roadmap Tier B: "Safety/Jailbreak Classifier — distill from
  gpt-oss-safeguard into KAN (2000x smaller)." This is the implementation path.
- **Concrete experiment:** Train KAN-based jailbreak classifier on EORM hidden state
  features (layers 8-16 of Qwen3.5-0.8B). Dataset: 100 adversarial + 100 benign code
  prompts. Target: AUC >= 0.90. If reached: file as Tier 0h (pre-generation safety gate).
- **When to incorporate:** Milestone 2026.04.59 — Phase 3 new research (Exp 775).

### LLM-JEPA — Large Language Models Meet JEPA (arXiv 2509.14252) ⭐ Phase 3
- **Paper:** arXiv 2509.14252 (September 2025)
- **What:** JEPA applied to LLMs: predicts future embeddings rather than next tokens.
  Joint embedding predictive architecture for text outperforms next-token prediction
  on both fine-tuning and pretraining tasks. Enables pre-generation trajectory prediction.
- **Relevance to Carnot:** JEPA v19 (Exp 770) trains a multi-step predictor on real
  violation data. LLM-JEPA extends this: instead of predicting "will this step be
  violated?" from text embeddings, predict the trajectory of the solution embedding
  and diverge early. This is the Phase 3 path to pre-generation guided decoding.
  Cite as motivation for Tier 3 predictive verification research direction.
- **When to incorporate:** Phase 3 foundation model research, after Tier 3 OOD AUC > 0.75.

### ARM↔EBM Bijection — Autoregressive Models are Secretly EBMs (arXiv 2512.15605) ⭐ FOUNDATIONAL
- **Paper:** arXiv 2512.15605 (December 2025)
- **What:** Establishes exact theoretical bijection between autoregressive language models
  and energy-based models. Every ARM implicitly defines an energy landscape; every EBM
  can be viewed as an implicit ARM. Unlocks EBM theory for LLM analysis.
- **Relevance to Carnot:** FOUNDATIONAL. This paper provides the theoretical basis for
  Carnot's entire approach: we are applying EBM inference (energy minimization, constraint
  satisfaction) to the implicit energy landscape of autoregressive LLMs. Every Carnot
  result implicitly relies on this bijection. Must be cited in all Carnot publications
  as the theoretical motivation for energy-guided decoding and constraint verification.
  Already filed in architecture.md as ARM-EBM bijection (2512.15605).
- **When to incorporate:** Already incorporated conceptually; cite explicitly in paper.

### MetaJuLS — Adaptive Constraint Propagation via Meta-RL (arXiv 2601.00095)
- **Paper:** arXiv 2601.00095 (January 2026)
- **What:** MetaJuLS learns universal constraint propagation policies via graph attention
  networks + meta-reinforcement learning, applicable across languages/tasks without
  retraining. 1.5-2.0x speedup vs GPU-optimized baselines while maintaining accuracy.
- **Relevance to Carnot:** Carnot's constraint memory is currently hand-tuned per task
  (arith, code, logic). MetaJuLS's approach could auto-learn constraint propagation
  policies that generalize across task types. Most relevant after Tier 2 memory is
  validated (Exp 761 showed precision 0.52->1.0 in 10 sessions). Meta-RL policy would
  accelerate constraint addition without requiring 3-session warm-up.
- **When to incorporate:** Milestone 2026.04.60+ after Tier 2 memory is stable.

### KV260 Open-Source Synthesis via nextpnr-xilinx
- **Finding:** Exp 758 (Milestone 2026.04.58) confirmed Yosys synthesizes ising_sampler_v2.v
  to 2821 LUTs, 2237 DFFs, 0 errors. The nextpnr-xilinx project provides open-source
  place-and-route for Xilinx Series 7 FPGAs, compatible with the KV260's XCZU5EV device.
  With Yosys+nextpnr, the full open-source synthesis flow is now potentially achievable
  without installing the full 80GB Vivado IDE.
- **Relevance to Carnot:** KV260 synthesis has been blocked for 6 consecutive milestones
  (Vivado not installed). nextpnr-xilinx provides an alternative that is pip-installable
  (python-prjtrellis or similar). After Yosys synthesis (Exp 758 confirmed), nextpnr is
  the only remaining step to generate a bitstream.
- **Concrete experiment:** Install nextpnr-xilinx (or prjtrellis-xilinx). Run P&R on the
  Yosys netlist from Exp 758. If bitstream generated: flash to KV260 and test hardware.
- **When to incorporate:** Milestone 2026.04.59 — Phase 4 hardware (Exp 776).

### Cognometry / Styxx — Empirical Cognitive-State Quantification (Fathom / Darkflobi)
- **URL:** https://fathom.darkflobi.com/cognometry
- **What:** Empirical framework ("cognometry") that classifies an LLM's internal
  cognitive state — refusal, confabulation, reasoning, drift — from residual-stream
  activations and logprob trajectories. Pooled logistic regression over 9 signals,
  no per-domain tuning. Ships a runtime instrument called **Styxx** with `@trust`
  decorator and `styxx.gate()` pre-flight classifier. Three empirical "laws":
  observable (AUC 0.998 on HaluEval-QA), transferable (cosine 0.464 within model
  family, 0.043 across vendors), steerable (refusal flipped 97% -> 17% via
  multi-position residual-stream injection). No formal mathematics; empirical and
  operational.
- **Relevance to Carnot:** Complementary, not overlapping. Cognometry reads
  *internal state* (white-box, needs residuals + logprobs); Carnot reads the
  *output* (black-box, any API LLM). A mature safeguard stack can use both.
  Four actionable takeaways:
  1. Logprob trajectories are black-box-accessible via most LLM APIs, so a
     reduced version of the Cognometry feature set could slot into the
     generative-time safety gate and the prompt-injection KAN v3 (currently
     AUROC 0.9078) without needing model-internals access.
  2. The cross-vendor transferability finding (cos 0.043) is a direct warning
     about our AUROC 0.9078 claim — the classifier was distilled from
     GPT-OSS-Safeguard and its cross-vendor performance is untested. A
     non-OSS-family held-out split is required before any publication gate.
  3. The 97% -> 17% steerability result strengthens the threat model in
     `conductor-self-protection-safeguard.md`: attackers with partial model
     access can actively shape generation, not just inject text. Reinforces
     our "treat all incoming search/tool/retrieved content as untrusted"
     stance.
  4. The `@trust` / `styxx.gate()` ergonomics is directly comparable to the
     surface we're drafting for `VerdictRecord` (issue #3) and `budget_ms`
     (issue #2). Worth reading before we finalize our API.
- **What NOT to adopt:** Cognometry deliberately skips formal mathematics.
  Carnot's invariant is the opposite — the energy function is ground truth
  and unhackable. Keep that stance; don't drift toward purely empirical
  classifiers.
- **Honest limitations they flagged:** reading comprehension AUC 0.424,
  financial arithmetic 0.492, partial cross-vendor transfer, untested above
  3B parameters.
- **When to incorporate:** Milestone 2026.04.63 — fold logprob-trajectory
  features into the `generative-time-safety-gate` proposal, and require
  cross-vendor validation for the prompt-injection classifier before any
  publication gate.

### UvA Deep Energy Models tutorial
- **Repo:** https://github.com/phlippe/uvadlc_notebooks (tutorial 8 — Deep Energy Models)
- **What:** Classroom-style PyTorch notebook that walks through the full EBM
  training cycle — contrastive divergence, Langevin sampling, mode coverage
  diagnostics — on an image-generation toy task. The pedagogy is explicit:
  every step of the CD-k update is derived in the markdown cells.
- **Relevance to Carnot:** Useful reference for onboarding engineers who are
  new to EBMs (the "verbose layman explanations" style we require in Carnot
  code mirrors how this notebook teaches). It's also a cross-check:
  our JAX training loop in `python/carnot/training/` should produce the
  same loss trajectories on the same toy problem as this notebook's PyTorch
  loop, which is a low-effort sanity check before chasing bugs elsewhere.
- **When to incorporate:** Reference material only; link from
  `docs/usage-guide.md` for readers coming from a PyTorch background.

### Equilibrium Matching (EqM) — advanced EBM training
- **Repo / page:** https://energy-based-model.github.io/
- **What:** A training objective that avoids the high-variance gradients of
  contrastive divergence by matching equilibrium distributions directly.
  Claimed to train more stably than CD-k and produce sharper modes.
- **Relevance to Carnot:** Carnot's Boltzmann and Gibbs tiers currently
  train via CD-k / denoising score matching. EqM is a candidate third
  training algorithm, particularly for the Boltzmann tier where CD-k's
  variance has historically caused long runs to collapse. Worth a
  side-by-side experiment before Phase 3 — if EqM converges more reliably
  on held-out energy, it's a better foundation for the self-learning loop.
- **When to incorporate:** Consider as a third training-algorithm option
  alongside CD-k and DSM, scheduled as a dedicated experiment when the
  Boltzmann tier's CD-k runs are next revisited.

### Extropic XTR-0 / X0 chips — thermodynamic sampler hardware
- **URL:** https://extropic.ai/writing/inside-x0-and-xtr-0
- **What:** Extropic's first public hardware designs. X0 is an analog
  probabilistic compute chip; XTR-0 is the development board that exposes
  X0's sampling primitives via a host-compatible interface. Together they
  implement p-bit-style Ising sampling at orders of magnitude lower energy
  than a GPU implementation of the same sampler.
- **Relevance to Carnot:** Direct Phase 2 hardware target. Our Ising tier
  and `SamplerBackend` protocol were deliberately designed to accept a
  thermodynamic backend alongside CPU and FPGA, and XTR-0 is the
  reference-quality path. The KV260 FPGA work is a stepping stone;
  XTR-0 is the production hardware Carnot aims to run on once Extropic
  opens up the SDK.
- **When to incorporate:** Monitor Extropic releases. When their SDK
  stabilises and we have a concrete API surface, add an `XtrBackend`
  implementation of `SamplerBackend` alongside `FpgaBackend`.

### mini-ebm — minimal educational EBM implementation
- **Repo:** https://github.com/yataobian/mini-ebm
- **What:** A single-file PyTorch EBM that fits on a screen. Deliberately
  stripped-down — no abstractions, no tiers, no MCMC options beyond basic
  Langevin. Intended purely as a teaching artifact.
- **Relevance to Carnot:** Best-in-class reference for the "what is an EBM
  actually doing?" conversation. When onboarding contributors or explaining
  the core loop to someone unfamiliar with energy-based modelling, this
  is the link to send. It also makes a good sanity-check harness — if we
  can reproduce its loss curve with Carnot's Ising tier on a matched
  toy problem, we know our plumbing is not wildly off.
- **When to incorporate:** Reference material only; link from
  `docs/usage-guide.md` and from onboarding material for new contributors.

### Recursive Language Models (RLMs) — long-context via recursive self-call
- **Paper:** arXiv 2512.24601 (December 2025) — Zhang, Kraska, Khattab
- **What:** Lets an LLM treat a long prompt as an environment and
  programmatically examine, decompose, and recursively call itself over
  snippets. Authors post-train **RLM-Qwen3-8B**. Headline numbers: process
  inputs up to **100x beyond model context windows**, +28.3% average over
  base Qwen3-8B on long-context benchmarks, approaching GPT-5 quality at
  comparable cost.
- **Relevance to Carnot:** Two scaling bottlenecks line up with the RLM
  pattern. (1) `LLMConstraintExtractor` truncates or silently drops claims
  when a reasoning chain exceeds the extracting model's window — recursive
  decomposition along reasoning-step boundaries lets us recover the
  dropped claims at the cost of more LLM calls. (2) `verify_stream` (issue
  #7) was drafted with implicit consumer-side decomposition; RLM makes
  the decomposition primitives explicit on the producer side. The two
  designs need to be reconciled before `verify_stream` ships. Energy-
  as-ground-truth is preserved — each recursive snippet's claims still
  go through the same Z3/Hypothesis/EBM verifier stack.
- **NOT for:** the generative-time safety gate (RLM recursion multiplies
  LLM cost, incompatible with the hard latency budgets in issue #2
  `budget_ms`); hallucination detection as such (RLM is a scale technique,
  not a truthfulness signal).
- **When to incorporate:** Milestone 2026.04.64 — full design checked in
  at `openspec/change-proposals/recursive-extractor-and-verify-stream-alignment.md`.

## 2026-04-24 arxiv Scan (Milestone 2026.04.63 Planning)

### AgentAuditor: Multi-Agent LLM Reasoning Tree Auditing Outperforms Majority Vote
- **Paper:** arXiv 2602.09341 (February 2026)
- **What:** Audits reasoning trees produced by multiple LLM agents by searching for
  divergence/agreement points across agent traces. Identifies "confabulation consensus" where
  agents converge on a shared wrong rationale. Achieves +5% absolute improvement over majority
  voting and LLM-as-Judge baselines on factual verification benchmarks.
- **Relevance to Carnot:** Directly addresses RETRO-ARBITER-FLAT-ENERGY from Exp 817
  (MultiAgentArbiter accuracy=0.33, all scores=0.0). Carnot's arbiter currently assigns
  identical energy to all agent responses because IsingConstraintInjector has a sign error.
  Once the energy sign is fixed (Exp 819), AgentAuditor's consensus detection logic can
  supplement energy ranking: when all agents converge (low energy divergence), apply a
  consensus penalty that rewards agents with distinct correct reasoning over shared wrong
  reasoning.
- **Concrete experiment:** Exp 822 — MultiAgentArbiterFixV2: incorporate AgentAuditor consensus
  detection as a tie-breaking signal when all agent energies are within 0.01 of each other.
  Test on 6 synthetic math debate scenarios + 6 adversarial scenarios where wrong answer
  is the majority. Target: arbiter_accuracy >= 0.80.
- **When to incorporate:** Milestone 2026.04.63 — Phase 1 arbiter fix (Exp 822).

### From Mathematical Reasoning to Code: Generalization of Process Reward Models
- **Paper:** arXiv 2506.00027 (May 2026)
- **What:** Analyzes cross-domain generalization of process reward models: trains PRMs on
  math (GSM8K, MATH-500) and evaluates on code generation (HumanEval). Shows PRMs transfer
  with moderate degradation (~8% AUC drop), with MCTS being most effective for abundant
  compute and Best-of-N for resource-limited scenarios.
- **Relevance to Carnot:** Carnot's JEPA has been trained and evaluated exclusively on GSM8K
  arithmetic. JEPA v22 (ood_auc=0.5 after RA-PRM) fails cross-domain because all training
  data comes from one distribution. This paper quantifies the baseline cross-domain degradation
  for PRMs, giving Carnot a comparison target. Also validates JEPA as a PRM-class model that
  should transfer — if Carnot's JEPA is below the published transfer degradation baseline,
  the data curation is the problem, not the architecture.
- **Concrete experiment:** Exp 826 — PRMCrossDomainBenchmark: evaluate JEPA v23 on GSM8K
  (in-distribution), HumanEval code steps (from Exp 820 results), and ARC-Challenge planning
  steps. Compare against the ~8% AUC cross-domain degradation baseline from this paper.
  CPU-only (uses stored CoT step results, no live GPU needed).
- **When to incorporate:** Milestone 2026.04.63 — Phase 2 cross-domain benchmark (Exp 826).

### Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning
- **Paper:** arXiv 2601.17223 (January 2026)
- **What:** Distinguishes verifiable outcome rewards (binary final answer check) from verifiable
  process rewards (step-by-step formal verification). Demonstrates that process rewards enable
  finer-grained error detection and improve PRM calibration on math + code tasks. The key
  insight: formal verifiability at the step level creates interpretable certificates that
  practitioners can audit, unlike scalar reward scores.
- **Relevance to Carnot:** Carnot's IsingEBM already produces step-level energy scores, but
  they lack formal verifiability certificates. This paper motivates adding a "step certificate"
  field to VerificationResult: which step(s) had high energy, what constraint was violated,
  and what Z3/SymCode verdict supports the violation claim. The certificate makes Carnot's
  output auditable beyond the scalar energy score, directly addressing the credibility gap
  for the enterprise market (Tier B: Compliance Checker).
- **Concrete experiment:** Exp 826 (joint with above) — add certificate output to JEPA v23
  evaluation: for each OOD step, emit a VerificationCertificate(step_id, energy_delta,
  constraint_type, z3_verdict, confidence). Evaluate certificate precision on known Z3-verified
  FoVer steps.
- **When to incorporate:** Milestone 2026.04.63 — Phase 2 cross-domain benchmark (Exp 826).

### Semantic Energy: Detecting LLM Hallucination Beyond Entropy
- **Paper:** arXiv 2508.14496 (August 2025)
- **What:** Combines semantic sentence clustering with a Boltzmann-inspired pairwise energy
  distribution. Coherent responses form tight semantic clusters (low energy); hallucinated
  responses with contradictory sentences produce near-zero pairwise kernel values (high energy).
  Outperforms entropy-based hallucination detection methods.
- **Relevance to Carnot:** Provides a new orthogonal verification signal — semantic coherence
  energy — that is independent of all existing tiers (logit-based, latent-space, thermodynamic,
  symbolic). Adds breadth to the cascade architecture at negligible cost (TF-IDF embeddings,
  no GPU required). Forms the basis for Tier 0f SemanticEnergyProbe.
- **Concrete experiment:** Exp 852 — SemanticEnergyProbe Tier 0f: implement pairwise Boltzmann
  semantic energy over sentence clusters as advisory signal. Wire in VerifyRepairPipeline
  between Tier 0e (HalluField) and Tier 0h (Jailbreak). Target: AUC_synthetic > 0.70.
- **When to incorporate:** Milestone 2026.04.65 — Phase 5 new capability (Exp 852).

### Gibbs Warm-Start Convergence Theory for Bayesian Models
- **Paper:** arXiv 2304.06993 (Dimension-free Mixing Times for Gibbs Samplers via Warm-Start)
- **What:** Proves that Gibbs samplers initialized from the limiting distribution (or a good
  approximation thereof) achieve dramatically faster mixing than cold-start initializations.
  The mean-field fixed point s_i = sign(h_i) is a tractable one-step approximation for the
  limiting distribution when biases dominate couplings.
- **Relevance to Carnot:** Carnot's MultiAgentArbiter has been non-functional (accuracy_standard
  = 0.0) for three consecutive milestones because the Gibbs sampler is cold-started from zero
  with only N_sweeps=10. The measured "energies" are initialization noise. Warm-start from
  sign(h_i) + 500 burn-in sweeps is the theoretically-grounded fix.
- **Concrete experiment:** Exp 846 — Arbiter Gibbs Warm-Start v3: implement GibbsWarmStart
  protocol (sign(h_i) init + 500 burn-in), apply to MultiAgentArbiter. Target:
  accuracy_standard >= 0.67, energy magnitudes in [-10, +10] range (vs current [-0.07, +0.14]).
- **When to incorporate:** Milestone 2026.04.65 — Phase 2 arbiter fix (Exp 846).

### KANELÉ: Kolmogorov-Arnold Networks for FPGA LUT Evaluation
- **Paper:** arXiv 2512.12850 (December 2025)
- **What:** First systematic KAN-to-FPGA synthesis flow with quantization and pruning
  co-optimization. Achieves compact, high-throughput KAN architectures on FPGA fabric.
  Uses LUT-based spline approximation for efficient on-chip KAN evaluation.
- **Relevance to Carnot:** Carnot's KAN energy tier (KAEMEnergy, Exp 447) is a natural
  candidate for FPGA acceleration. KANELÉ provides the synthesis methodology to take
  a JAX-trained KAN model and map it to iCE40/KV260 fabric. This is a concrete path
  to hardware-accelerated constraint checking (Tier 1 self-learning hardware path).
  Phase 3 foundation model will need KAN-on-FPGA for real-time energy evaluation.
- **Concrete experiment:** Future milestone — KAN FPGA synthesis: use KANELÉ flow to
  synthesize the KAEMEnergy model (Exp 447) onto KV260 fabric alongside the Ising sampler.
  Joint deployment: Ising samples from FPGA, KAN energy is evaluated on FPGA.
- **When to incorporate:** After iCE40 N=16 bitstream confirmed (Exp 851). Target .66 or .67.

### Rethinking Reward Models for Multi-Domain Test-Time Scaling
- **Paper:** arXiv 2510.00492 (October 2025)
- **What:** First unified evaluation of reward models across 14 domains. Finds that
  discriminative outcome reward models (DisORM) are competitive with process reward models
  (DisPRM) when aggregated across diverse domains. Domain-specific calibration matters more
  than model architecture for multi-domain deployment.
- **Relevance to Carnot:** Establishes the multi-domain evaluation baseline that Carnot's
  JEPA needs to beat. The 14-domain evaluation protocol provides a rigorous test harness.
  Carnot's current 4-domain evaluation (GSM8K, HumanEval, ARC, SVAMP) is a subset.
  The finding about calibration-vs-architecture suggests that DreamPRM domain reweighting
  (what Exp 844 uses) is the right intervention — calibration over architecture changes.
- **Concrete experiment:** Exp 844 (JEPA v24b) implicitly tests this claim: if domain
  reweighting (DreamPRM) fixes the SVAMP collapse, it confirms that calibration is the
  critical variable for JEPA multi-domain performance.
- **When to incorporate:** Milestone 2026.04.65 — Exp 844 evaluation protocol.

### Decomposing Large-Scale Ising Problems on FPGAs
- **Paper:** arXiv 2602.15985 (February 2026)
- **What:** Edge-class FPGA (XC7A35T, ~33K LUTs): 87% LUT utilization, 80% BRAM at 100 MHz,
  0.73W. Hybrid decomposition approach for large-scale Ising problems. Shows practical FPGA
  resource budgets for small-N Ising designs (N=16 < 1000 LUTs, well within XC7A35T/HX8K).
- **Relevance to Carnot:** Direct reference for iCE40 N=16 Ising synthesis (Exp 851).
  The paper's resource estimates for XC7A35T are comparable to iCE40 HX8K (both ~7K LUTs
  effective). Confirms N=16 at ~1000 LUTs is feasible. The hybrid decomposition approach
  could extend Carnot's N=16 prototype to N=64 without exceeding FPGA budget.
- **Concrete experiment:** Exp 851 — iCE40 N=16 Ising bitstream. The paper's LUT estimates
  validate the design hypothesis: N=16 should synthesize at ~1000 LUTs vs N=32's 3952 LUTs.
- **When to incorporate:** Milestone 2026.04.65 — Phase 5 FPGA experiment (Exp 851).

### CHARM: Calibrating Reward Models With Chatbot Arena Scores
- **Paper:** arXiv 2504.10045 (April 2025)
- **What:** Post-hoc reward model calibration using Elo-score comparisons. Addresses model
  preference bias — RMs that systematically overvalue certain output styles. Calibration via
  a learned scalar transformation of RM scores anchored to human preference data.
- **Relevance to Carnot:** MultiAgentArbiter's energy scores need calibration beyond just
  L2-normalization and Z-scoring. Once Gibbs warm-start (Exp 846) produces meaningful energy
  magnitudes, CHARM-style calibration could further improve arbitration accuracy by anchoring
  energy rankings to known-correct human preference data (the MATH or HumanEval ground truth).
- **Concrete experiment:** Future milestone — post Exp 846: if accuracy_standard reaches 0.67+
  but adversarial accuracy remains inconsistent, apply CHARM calibration using HumanEval
  ground truth as preference anchors.
- **When to incorporate:** After Exp 846 confirms warm-start works. Target .66.

## 2026-04-25 arxiv Scan (Milestone 2026.04.68 Planning)

### Hallucination Detection in LLMs Using Spectral Features of Attention Maps
- **Paper:** arXiv 2502.17598 (February 2025)
- **What:** Uses eigenvalues of the graph Laplacian of attention maps to predict hallucinations.
  The intuition: attention maps in hallucinating passages have flatter eigenvalue spectra
  (more diffuse attention) than factually correct ones. Computes spectral energy as
  E_spectral = sum_i lambda_i * log(lambda_i + epsilon) and trains a lightweight linear probe.
  Achieves F1 > 0.82 at less than 0.5ms overhead per token on Llama/Qwen families.
- **Relevance to Carnot:** The cascade currently lacks a spectral-geometry tier. All existing
  Tier 0 probes operate on token probabilities (Tier 0b, 0e) or CoT step boundaries
  (Tier 0c, 0g). A spectral attention probe is geometrically orthogonal to all existing tiers
  — it measures the spatial distribution of attention weight, not its magnitude. The 0.5ms
  overhead makes it feasible as Tier 0h advisory. Specifically: compute spectral energy over
  each CoT step's attention map, flag is_spectrally_diffuse in VerificationCertificate.
  The bigram proxy (as used in NUP Probe v4 and HalluSAE) can approximate the Laplacian
  without requiring actual attention map access — makes it CPU-only.
- **Concrete experiment:** Exp 885 — SpectralAttentionProbe (Tier 0h): implement a bigram-proxy
  Laplacian energy probe over CoT text (token co-occurrence as adjacency). Compute spectral
  entropy E = -sum lambda_i * log(lambda_i). Train linear probe on 50 synthetic CoT pairs.
  Target: AUC > 0.70 as advisory Tier 0h. CPU-only implementation.
- **When to incorporate:** Milestone 2026.04.68 — Phase 3 new probes (Exp 885).

### Constrained Decoding for Code Generation via AST Token Masking
- **Paper:** arXiv 2508.15866 (August 2025)
- **What:** Implements a context-sensitive AST parser that constraints LLM token generation
  at each step. For each partial code generation, a parser computes the set of valid next
  tokens given the current AST state, masking the logit vector to prevent syntactically
  invalid tokens from ever being sampled. Achieves 100% syntactic correctness on Python
  output with less than 5% latency overhead.
- **Relevance to Carnot:** Code repair currently runs post-hoc: Carnot waits for the LLM to
  generate a syntactically invalid or semantically wrong response, then tries to fix it.
  Constrained decoding is preventive: violations never enter the response because they are
  masked out at generation time. Combined with Carnot's energy-based post-hoc verification,
  this creates a two-layer defense: prevent syntactic violations (constrained decoding) then
  catch semantic violations (Carnot). Expected effect: CodeExtractor false-positive rate
  drops significantly because generated code is always syntactically valid, narrowing the
  search space for semantic errors.
- **Concrete experiment:** Exp 886 — ConstrainedDecodingPreFilter: implement a lightweight
  Python AST validator that masks clearly-invalid next tokens during generation (e.g., reject
  tokens that would produce syntax errors in a partial AST). Apply as a pre-filter before
  VerifyRepairPipeline on HumanEval. Measure: FP rate reduction, pass@1 delta.
- **When to incorporate:** Milestone 2026.04.68 — Phase 3 new capabilities (Exp 886).

### Process Reward Models Meet Planning: Automatic Step-Level Reward Data Generation
- **Paper:** arXiv 2604.17957 (April 2026)
- **What:** Automatically generates step-level reward data for PRMs by running a planning
  model (MCTS-style) and recording which intermediate steps lead to correct vs incorrect
  final answers. This provides dense supervision at every reasoning step without human
  annotation. Reduces data requirements for PRM training by 10x vs. human-annotated datasets.
- **Relevance to Carnot:** JEPA's training corpus has only 57 real FoVer-labeled pairs
  (from live GPU runs). That is far too few for OOD generalization. This paper's automatic
  data generation approach could expand the corpus to 500+ pairs without additional human
  effort: run Qwen3.5-0.8B on 200 GSM8K questions, use the ground-truth answer to label
  each intermediate step as correct/incorrect, then use those labels as JEPA training targets.
  This synthetic corpus would give VJEPA v2 enough data to generalize OOD.
- **Concrete experiment:** Applies directly to Exp 883 (VJEPA v2 training) — generate
  synthetic step-level labels from Qwen3.5-0.8B + GSM8K ground truth. Augments FoVer corpus.
- **When to incorporate:** Milestone 2026.04.68 — Phase 2 VJEPA training (Exp 883).

## 2026-04-26 arxiv Scan (Milestone 2026.04.69 Planning)

### Latent Veracity Inference for Identifying Errors in Stepwise Reasoning
- **Paper:** arXiv 2505.11824 (May 2025)
- **What:** Assigns latent correctness variables to each reasoning step and proposes amortized
  veracity inference for zero-shot error detection. Treats step-level verification as Bayesian
  posterior estimation over latent truth states: P(correct | partial_chain) updated at each step.
  Achieves strong zero-shot error detection without task-specific labels.
- **Relevance to Carnot:** VJEPA v2 (Exp 883, ood_auc=0.9211) predicts violation probability but
  treats each step as independent. Latent Veracity Inference would make the predictor causal:
  P(step_i correct | step_1..step_{i-1}). This is the theoretical grounding for the VJEPA Live
  Streaming Filter (Exp 894) — posterior updates propagate across CoT steps, catching compounding
  errors earlier than step-independent prediction.
- **Concrete experiment:** Enhancement to Exp 894 (VJEPA Live Streaming Filter): add running
  posterior update across CoT steps. P(violation | step_i) = VJEPA(step_i) * P(violation | step_{i-1}).
  Compare streaming posterior vs independent step-by-step VJEPA predictions.
- **When to incorporate:** Milestone 2026.04.69 — Phase 1 GPU (Exp 894).

### Controlled LLM Decoding via Discrete Autoregressive Biasing
- **Paper:** arXiv 2502.03685 (February 2026)
- **What:** Energy-based decoding defines target distributions through energy functions combining
  multiple constraints during token generation. At each generation step, candidate tokens are
  scored by an energy function and logits are biased toward low-energy continuations. Achieves
  constrained generation without fine-tuning via discrete Langevin-style perturbation.
- **Relevance to Carnot:** Carnot currently verifies AFTER generation (post-hoc). This paper
  enables verification DURING generation — the energy function guides token selection away from
  constraint violations before they appear. Directly applicable to VJEPA Live Streaming (Exp 894):
  instead of flagging violations after each step, bias the LLM's next-token probabilities using
  VJEPA's violation energy as a soft constraint on the generation beam.
- **Concrete experiment:** Extension of Exp 894: after VJEPA predicts violation_prob > 0.5 at
  step i, bias logits for the next generation step using -violation_prob as a negative energy
  bonus on "repair" tokens (numbers, recalculations). Measure: does energy-biased generation
  produce fewer violations than flag-and-repair?
- **When to incorporate:** Milestone 2026.04.69 — Phase 1 GPU (Exp 894).

### Estimation Verification for Math Word Problems (SVAMPClean)
- **Paper:** arXiv 2509.18565 (September 2025)
- **What:** Two-stage verification for math word problems: (1) LLM generates an equation from
  the decomposed problem, (2) LLM independently estimates the answer. Equation and estimate are
  compared via symbolic solver. If they disagree: iterative rectification. Achieves SOTA on SVAMP
  and introduces SVAMPClean (50 corrected ambiguous SVAMP questions). Does not require multi-step
  CoT — works on single-step word problem solutions.
- **Relevance to Carnot:** SVAMP AUC=0.125 (Exp 872) is near-random. Root cause hypothesis:
  SVAMP questions are single-step word problems ("how many more?") that don't have labeled
  multi-step CoT chains — FoVer labeling and VJEPA both require step sequences. The estimation
  verification approach from this paper bypasses step-labeling entirely: extract equation from
  answer, verify against independent estimate. This could bring SVAMP AUC from 0.125 to > 0.60
  without requiring any multi-step CoT structure.
- **Concrete experiment:** Exp 896 — SVAMPEstimationVerifier: implement equation extraction +
  estimation comparison for SVAMP format. Use as training signal for VJEPA v3 SVAMP corpus.
  Also file SVAMPClean as the standard evaluation split.
- **When to incorporate:** Milestone 2026.04.69 — Phase 2 SVAMP fix (Exp 896).

### FOREVER: Forgetting Curve-Inspired Memory Replay for LLM Continual Learning
- **Paper:** arXiv 2601.03938 (January 2026)
- **What:** Aligns memory replay schedules with a "model-centric notion of time" using the
  magnitude of optimizer updates. Rather than fixed-interval replay, FOREVER replays past data
  most intensively when the model is changing fastest (large gradient steps). The forgetting
  curve decays memory importance proportionally to elapsed model-update magnitude.
- **Relevance to Carnot:** Tier 1 Lagrange adaptive weights (Exp 862, fr11_self_learning_confirmed)
  update at each verification step. Constraints that haven't fired in many queries accumulate
  positive Lagrange weight indefinitely. FOREVER-inspired weight decay: λ_i decays when VJEPA
  violation signals don't involve constraint i for N queries. This prevents stale constraints
  from biasing the energy function and enables automatic pruning of domain-specific constraints
  when the query distribution shifts.
- **Concrete experiment:** Exp 897 — ForgettingCurveScheduler: λ_i *= exp(-decay_rate *
  (t - last_fired_i)) where last_fired_i is the last query index where constraint i fired.
  decay_rate tuned so constraints not fired in 100 queries drop to 50% weight. 10-session relay.
- **When to incorporate:** Milestone 2026.04.69 — Phase 2 self-learning (Exp 897).

### DRIFT: Detecting Representational Inconsistencies for Factual Truthfulness
- **Paper:** arXiv 2601.14210 (January 2026). Accepted to ACL 2026.
- **What:** Lightweight linear probe trained on pre-generation activations of frozen LLM detects
  factual truthfulness without any model modification. Tracks "representational drift" — the
  difference between activation patterns when the model "knows" a fact vs. when it generates an
  incorrect claim. Uses token patching to identify which attention layers encode truthfulness.
- **Relevance to Carnot:** HiddenStateHalluProbe (planned for .68, deferred) uses MLP on final-
  layer hidden states. DRIFT provides a stronger prior: early layers (e.g., layers 4-8) may
  encode truthfulness better than final layers. Multi-layer linear probe ensemble (consistent
  with arXiv 2604.13386 which shows +29-78% AUROC from ensembling) could improve Carnot's probe
  AUC to > 0.90. Frozen probe — no fine-tuning, NPU-deployable.
- **Concrete experiment:** Exp 899 — DRIFTProbe: train 3-layer linear probe on Qwen3.5-0.8B
  residual stream activations at layers 4, 8, 12. Train on 57 real FoVer pairs + 150 synthetic.
  Compare: single-layer vs multi-layer ensemble. Target: AUC > 0.90.
- **When to incorporate:** Milestone 2026.04.69 — Phase 3 research (Exp 899).

### Draft-Conditioned Constrained Decoding for Structured Generation
- **Paper:** arXiv 2603.03305 (March 2026)
- **What:** Two-step, training-free approach: generate an unconstrained draft first, then apply
  constrained decoding conditioned on the draft. The draft functions as semantic planning before
  structural enforcement. Improves structured accuracy by up to 24 percentage points by reducing
  cases where the constraint mask finds no valid continuations.
- **Relevance to Carnot:** ConstrainedDecodingPreFilter (Exp 886) applies AST-based token masking
  post-hoc. Draft-conditioned approach would work differently: before full generation, generate a
  1-sentence "draft answer" with the expected structure (e.g., "approximately 15 items"). Use this
  draft to constrain what the full response should say, catching structural violations before the
  full reasoning chain is generated. Reduces CodeExtractor FP rate by ensuring the code "scaffold"
  is valid before adding logic.
- **Concrete experiment:** Exp 900 — DraftConditionedVerifier: generate 1-sentence draft estimate
  before full CoT. Extract numerical/structural constraints from draft. Use as soft priors in the
  cascade (expected_answer_range feeds into ArithmeticExtractor threshold). Test on 50 synthetic
  GSM8K questions. Measure: constraint extraction rate vs without draft conditioning.
- **When to incorporate:** Milestone 2026.04.69 — Phase 3 research (Exp 900).

### Scalable Connectivity for Ising Machines: Dense to Sparse
- **Paper:** arXiv 2503.01177 (March 2025)
- **What:** Proposes systematic sparsification of dense Ising graphs by introducing copy nodes
  to limit the number of neighbors per spin to K_max (typically K_max=4 or 8). For N=64 spins
  with all-to-all coupling (N^2 edges), copy-node sparsification reduces to K_max*N edges with
  minimal quality loss. Provides theoretical quality bounds as a function of K_max.
- **Relevance to Carnot:** RETRO-INERTIA-SWEEPS-TARGET-MISSED: PIMI v3 parallel (Exp 889) may
  still miss 5x because N=8 has dense coupling (K=7 per spin). Copy-node sparsification would
  reduce effective K from 7 to K_max=4, improving per-sweep efficiency. Also directly relevant
  to the RETRO-ICE40-N16 issue (register expansion): sparse adjacency dramatically reduces
  BRAM/LUT usage for larger N.
- **Concrete experiment:** Exp 901 — PIMI Sparse v4: implement copy-node sparsification for N=8
  (K_max=4). Compare: dense N=8 PIMI vs sparse-N=8 PIMI. If sparse >= 5x sweep reduction: close
  RETRO-INERTIA. If still < 5x: retire PIMI approach to exclusion manifest.
- **When to incorporate:** Milestone 2026.04.69 — Phase 3 hardware (Exp 901).

### Efficient Probabilistic Ising Machines with Full Parallel Updates (PIMI)
- **Paper:** arXiv 2604.17109 — already referenced in .66 scan. Key detail missed:
  The 15-25x sweep reduction is achieved ONLY with truly parallel spin updates (all spins
  update simultaneously, not checkerboard). The inertia EMA (Exp 860 / Exp 876) is a
  necessary but not sufficient condition — it prevents oscillations so that parallel updates
  are stable. Without full parallelism, EMA alone gives only 2-3x reduction. Exp 876's
  failure (2x achieved, 5x missed) was precisely because it used checkerboard updates with
  EMA, not fully parallel updates with EMA. The v3 experiment must implement synchronous
  full-parallel update as the primary change.
- **Relevance to Carnot:** RETRO-INERTIA-SWEEPS-TARGET-MISSED root cause now diagnosed.
  The fix is architectural (parallel update scheduling), not parameter-based (alpha tuning).
- **Concrete experiment:** Exp 889 — iCE40 PIMI v3: implement truly parallel spin update
  (ALL spins flip in same clock cycle based on h_ema from PREVIOUS cycle — no checkerboard).
  EMA update uses h_ema_prev as input, h_ema_new as output in separate pipeline stages.
  Python simulation first: compare parallel vs checkerboard at same alpha. Then Verilog.
- **When to incorporate:** Milestone 2026.04.68 — Phase 5 hardware (Exp 889).


## Symbolic-KAN (arXiv 2603.23854, April 2026)

- **Title:** Symbolic-KAN: Augmenting Kolmogorov-Arnold Networks with Discrete Symbolic Node Labels
- **arXiv:** 2603.23854
- **Key idea:** Each KAN node is assigned a discrete symbolic label from a predefined vocabulary
  (ADD, MUL, CMP, EQ). The node's forward pass combines the symbolic function's output with a small
  learnable residual spline correction. Symbolic labels are updated via discrete search (argmin over
  vocabulary) every N gradient steps, making the learned constraint function fully interpretable.
- **Four node types:**
  - ADD: f(x,y) = x+y — checks additive relationships
  - MUL: f(x,y) = x*y — checks multiplicative relationships
  - CMP: f(x,y) = sign(x-y) — checks comparison direction
  - EQ:  f(x,y) = |x-y| — checks equality (low = equal)
- **Relevance to Carnot:** Exp 937 applied Symbolic-KAN to arithmetic constraint verification.
  AUC = 0.9344 vs standard KAN baseline 0.2208 (delta = +0.7136). Verdict: symbolic_kan_viable.
  Interpretability is the primary gain: each node announces its semantic constraint role.
- **Implementation:** python/carnot/models/symbolic_kan.py (REQ-MODEL-030, SCENARIO-MODEL-015).
- **When incorporated:** Exp 937 — Milestone 2026.04.26.

## 2026-04-26 arxiv Scan (Milestone 2026.04.73 Planning)

### ThinkPRM: Process Reward Models That Think
- **Paper:** arXiv 2504.16828 (April 2026)
- **What:** Generative process reward model that verifies each reasoning step by producing a
  verification chain-of-thought rather than a discriminative binary score. Trained with only
  1% of the process labels required by discriminative PRMs. Achieves 8% improvement over
  discriminative verifiers on GPQA-Diamond and 4.5% on LiveCodeBench. Outperforms LLM-as-a-Judge
  by 7.2% under equivalent token budgets. Demonstrated on ProcessBench, MATH-500, AIME '24.
- **Key insight:** Having the model explain WHY a step is correct/incorrect before scoring
  provides a much stronger gradient signal than binary labels. The verification CoT forces the
  model to internalize mathematical reasoning rather than learning pattern matching. Only 8K
  process labels needed (vs 800K in PRM800K).
- **Relevance to Carnot — RETRO-HEURISTIC-RPRM-FLAT-SIGNAL fix:**
  Exp 924 (R-PRM Tier 2.9) produced AUC delta=0 in heuristic mode — the "explain then score"
  pipeline used rule-based explanations instead of model-generated reasoning. ThinkPRM shows
  the explanations must come from the model itself, not rules. Implementing ThinkPRM for Carnot:
  pass each reasoning step to the LLM with the prompt "Verify this step: [step]. Generate
  a chain-of-thought explaining whether this step is correct, then output CORRECT or INCORRECT."
  Use the resulting token probabilities (P(CORRECT)) as the step score. Feed into Ising EBM
  as a learned prior on constraint satisfaction probability.
- **Decentralization note:** Works with any LLM, including local GGUF models. Does not require
  proprietary annotation (unlike PRM800K). Open approach — aligns with CLAUDE.md rule 1.
- **Concrete experiment:** Exp 945 — ThinkPRM Tier 2.9: implement generative CoT step verifier
  using Gemma4-E4B-it locally. Compare AUC vs Exp 924 heuristic baseline.
- **When to incorporate:** Milestone 2026.04.73 — Phase 3.

### E-MVL: Sparsified Ising Machine via Extraction-Type Majority Voting Logic
- **Paper:** arXiv 2604.04606 (April 2026)
- **What:** Quantum-inspired Ising machine algorithm implemented in digital FPGA logic.
  Sparsifies spin connectivity by an extraction-type majority voting logic (E-MVL), which
  mimics thermal spin dynamics while using primarily integer additions and logic operations.
  Handles up to 1600 spins (4x more than simulated annealing at 400). Achieves ~6x faster
  solution speed than simulated annealing on FPGA. Outperforms all competitors on
  Sherrington-Kirkpatrick model benchmarks.
- **Key insight:** Sparse connectivity (not dense coupling) is the key to FPGA efficiency.
  Rather than computing E_i = sum_j(J_ij * s_j) for all j, E-MVL computes only the majority
  vote over a subset of neighbors. This reduces logic operations from O(N^2) to O(N log N)
  and maps directly to FPGA LUT structures.
- **Relevance to Carnot — KV260 RTL path:**
  The current hardware/kv260/ising_sampler_v1.v (v2 also) uses dense connectivity, contributing
  to the N=128 LUT overflow (RETRO-072). E-MVL sparsification would reduce the LUT count from
  290K (overflow) to within budget for N=64-128 spins. E-MVL also enables larger spin counts on
  the KV260's 117K LUT budget. Should be applied to ising_sampler_v3_spec.md (already has
  inertia dynamics; add sparse connectivity on top).
- **Concrete experiment:** Exp 950 — E-MVL Sparsified Ising: implement sparse connectivity
  pattern in Python (CPU simulation), compare AUC and convergence speed vs dense Ising, then
  write the RTL spec update for v4 (sparse + inertia).
- **When to incorporate:** Milestone 2026.04.73 — Phase 5.

### IIPC: Iteratively Improved Program Construction for Math
- **Paper:** arXiv 2602.03950 (February 2026)
- **What:** Execution-driven math reasoning that converts word problems to Python programs,
  executes them, and feeds back execution errors (tracebacks) to the LLM for iterative repair.
  Combines programmatic chain-of-thought with execution feedback. Bridges code repair techniques
  to mathematical reasoning.
- **Key insight:** Unlike direct text-based math repair, IIPC provides EXTERNAL feedback
  (Python execution results and errors) that re-enters the model at the input layer. This
  sidesteps the topological problem identified in arXiv 2604.17121 — mathematical state
  computed in one forward pass cannot be retrieved in the next without external grounding.
  Programs provide that external grounding.
- **Relevance to Carnot — RETRO-MATH-REPAIR-MODEL-CEILING context:**
  Exp 930 failed because gemma-4-E4B-it has a 12% GSM8K baseline — no margin for repair.
  IIPC provides a parallel approach: instead of re-prompting the model to "think again," generate
  a Python program that computes the answer, execute it, check the result. The program execution
  provides a deterministic oracle that is independent of model capability. Even a small model
  that can't reliably compute 47+28 in text can often generate `print(47+28)` correctly.
- **Concrete experiment:** Consider for Milestone 2026.04.74 after SOTA math repair results
  are established. IIPC may outperform text-based repair for small models while text-based
  repair may work better for large models where mathematical reasoning is strong.
- **When to incorporate:** Milestone 2026.04.74 (deferred — see Exp 942-943 first).

### AdaDec: Uncertainty-Guided Adaptive Decoding
- **Paper:** arXiv 2506.08980 (June 2025)
- **What:** Pause-then-rerank mechanism for LLM decoding that triggers when Shannon entropy at
  a token position exceeds a learned model-specific threshold. At high-uncertainty positions,
  generates multiple candidate continuations and reranks them via lookahead scoring. Achieves
  consistent improvement on code generation benchmarks (+4.2pp HumanEval). Learns per-model
  uncertainty thresholds from a small calibration set (100 examples).
- **Relevance to Carnot:** Carnot's ThinkProbe (Exp 444) uses a fixed confidence threshold.
  AdaDec's per-model learned threshold is more accurate. Integration path: use Carnot's
  IsingEBM energy as the reranking score (instead of lookahead) — when the LLM's token entropy
  exceeds the threshold, generate K candidates and select the lowest-energy one. This combines
  AdaDec's pause-trigger with Carnot's energy-based selection.
- **Concrete experiment:** Consider for future milestone — energy-guided adaptive decoding
  combining AdaDec trigger with Carnot IsingEBM reranker.
- **When to incorporate:** Future milestone after ThinkPRM Tier 2.9 is established.

### Uncertainty-Aware Step Verification via CoT Entropy
- **Paper:** arXiv 2502.11250 (Feb 2025)
- **What:** Proposes augmenting generative process reward models (PRMs) with uncertainty
  quantification by measuring entropy over chain-of-thought verification tokens. When the
  verifier's own reasoning is high-entropy (uncertain), the step score is down-weighted and
  flagged for review. Achieves AUROC improvement of 0.04 on MATH step-level benchmarks vs
  ThinkPRM baseline by abstaining on ambiguous steps rather than forcing a verdict. The key
  insight: a verifier that knows when it does NOT know is more useful than one that always
  guesses. CoT entropy is measurable without additional supervision.
- **Relevance to Carnot:** ThinkPRM Tier 2.9 (Exp 945, AUROC=0.99) generates verification
  CoTs but does not quantify its own uncertainty. Adding CoT-entropy UQ to the Tier 2.9
  path could raise precision further by flagging low-confidence verdicts for escalation to
  Tier 3 (EBM-based) rather than making high-stakes decisions with marginal signal. The
  uncertainty signal also feeds naturally into Carnot's energy-based repair: if the verifier
  is uncertain, run an Ising MCMC repair rather than just rejecting.
- **Concrete experiment:** Milestone 2026.04.75+ — integrate CoT-entropy uncertainty
  into ThinkPRM Tier 2.9 after live GPU validation (Exp 954) establishes a baseline.
- **When to incorporate:** Milestone 2026.04.75 (ThinkPRM live baseline must come first via Exp 954).

### Divide-and-Conquer Neural Network Surrogates for Sparse Ising Sampling
- **Paper:** arXiv 2604.20701 (Apr 2026)
- **What:** Trains lightweight neural network surrogates to approximate sparse Ising energy
  landscapes, then uses divide-and-conquer decomposition to split a large sparse Ising problem
  into overlapping subproblems that each fit the surrogate's capacity. Achieves 20x MCMC
  speedup on sparse Ising problems (N=256, K=8 nonzero couplings per spin) vs standard MCMC
  with <0.5% distribution divergence (KL). The key insight: sparse coupling structure means
  subproblems are mostly independent; a small NN can approximate the marginal energy of each
  subproblem, and the decomposition recovers the joint without exponential blowup.
- **Relevance to Carnot:** Carnot's Ising sampler bottleneck (identified in E-MVL Exp 950)
  is primarily the O(N^2) dense coupling matrix. The divide-and-conquer approach maps
  directly to the E-MVL sparsified Ising (K=16) architecture — sparsity is already a design
  requirement of the v4 RTL spec (hardware/kv260/ising_sampler_v4_spec.md). On CPU/GPU,
  the 20x speedup would cut the current ~40ms Ising sample time to ~2ms, enabling real-time
  verification within LLM token generation latency. On KV260 FPGA, the NN surrogate could
  be implemented as a small fixed-point MLP.
- **Concrete experiment:** Consider for Milestone 2026.04.75+ after KV260 v4 RTL implementation
  (Exp 958) establishes the sparse baseline — NN surrogate can then be evaluated as a software
  complement to the hardware accelerator.
- **When to incorporate:** Milestone 2026.04.75 (Exp 958 KV260 v4 RTL must come first for baseline).

### iCKANs: Inelastic Constitutive KANs with Symbolic Regression
- **Paper:** arXiv 2602.17750 (Feb 2026)
- **What:** Extends Kolmogorov-Arnold Networks (KANs) to learn inelastic constitutive
  relationships in physical materials modelling. Uses symbolic regression on the learned
  KAN activation functions to recover closed-form constitutive equations from experimental
  stress-strain data. The key contribution is regularization for physical plausibility:
  thermodynamic consistency constraints (positive dissipation, convexity) are enforced as
  soft penalties during KAN training, which guides symbolic regression toward physically
  meaningful formulae. AUC-equivalent: recovered equations have <2% RMS error on held-out
  data while being interpretable symbolic expressions.
- **Relevance to Carnot:** Carnot's Symbolic-KAN (Exp 948, AUC=1.0 on 57 real FoVer pairs)
  uses B-spline activations without physics regularization. The iCKAN approach maps to
  logical-constraint regularization for Carnot's case: instead of thermodynamic consistency,
  enforce monotonicity (higher violation severity → higher energy), convexity (energy
  increases away from the constraint manifold), and symbolic parsimony (prefer shorter
  formulae that generalize better). These regularizers address the known failure mode where
  KANs overfit on small real-data sets (57 pairs is borderline). The iCKAN pattern also
  provides a path to symbolic-form EBM energy functions — energy(x) = f(x) where f is a
  discovered closed-form formula, enabling hardware-acceleratable energy computation.
- **Concrete experiment:** Milestone 2026.04.75 — iCKAN regularization for Symbolic-KAN
  Tier 3.1 after pipeline deployment (Exp 960) establishes the unregularized baseline.
- **When to incorporate:** Milestone 2026.04.75 (Exp 960 Symbolic-KAN deploy must come first).

## 2026-04-27 arxiv Scan (Milestone 2026.04.75 Planning)

### Optimal Abstractions for Verifying Properties of KANs (KAN-MILP)
- **Paper:** arXiv 2602.06737 (February 2026)
- **What:** Creates mathematical abstractions by replacing each KAN unit with a piecewise affine
  (PWA) function and encodes the verification problem as a Mixed Integer Linear Program (MILP).
  Dynamic programming at the unit level combined with knapsack optimization across the network
  finds the minimum number of PWA pieces needed for accurate abstractions. Uses Python 3.11 +
  Gurobi 12.0.1. Verified safety properties (input-output bounds, monotonicity, convexity)
  for KAN models trained on standard benchmarks; upfront analysis amortized over many queries.
- **Relevance to Carnot:** Carnot's KAN energy tier (KAEMEnergy, Symbolic-KAN) is a trust-critical
  component — a subtle spline bug could produce incorrect energy scores (high energy = correct,
  low energy = wrong) without any observable test failure. MILP-based KAN verification could
  catch three classes of Carnot bugs that standard testing misses: (a) monotonicity violations
  (energy should increase with violation severity), (b) convexity violations (energy landscape
  should be smooth near constraint boundaries), and (c) output-range violations (Ising energy
  should stay in [-N, N] for N spins). Verification overhead is one-time (analysis phase),
  then query-time is unchanged. Hardware path: MILP is CPU-bound; piecewise-affine abstraction
  is FPGA-native (already the structure of KAEMEnergy's bisection sampling).
- **Concrete experiment:** Exp 972 (Milestone 2026.04.75) — KAN Formal Verification via MILP:
  apply the PWA abstraction method to Carnot's UnivariateKAEMLayer and Symbolic-KAN checkpoint.
  Check: (1) monotonicity of energy w.r.t. violation indicator input, (2) output range bounds,
  (3) convexity near the constraint boundary hyperplane. Report: n_properties_verified,
  n_violations_found, verification_time_s.
- **When to incorporate:** Milestone 2026.04.75 (Phase 4 — after Symbolic-KAN deployed in Exp 968).

### PPSEBM: Progressive Parameter Selection EBM for Continual Learning
- **Paper:** arXiv 2512.15658 (December 2025)
- **What:** Integrates an Energy-Based Model (EBM) with Progressive Parameter Selection (PPS)
  to prevent catastrophic forgetting in continual NLP learning. PPS allocates distinct,
  task-specific parameter subsets for each new task; the EBM generates pseudo-samples from
  prior tasks to guide parameter selection. Outperforms state-of-the-art continual learning
  baselines across NLP benchmarks. Accepted IEEE International Conference on Big Data 2025.
- **Relevance to Carnot:** Carnot's Tier 2 cross-session memory (Exp 748) plateaus at session 2
  — no new templates added after the initial session, and Tier 2.1 JEPAReasonerProbe also
  stalled. Root cause: the constraint memory store does not distinguish which parameters are
  relevant to new vs old constraint types, leading to interference and saturation. PPSEBM's
  progressive parameter selection directly addresses this: when a new constraint type is
  detected (e.g., "arithmetic carry-forward errors"), allocate a fresh parameter slice to it
  rather than overwriting the existing constraint memory. EBM pseudo-sample generation ensures
  old constraint patterns are retained while new ones are learned. This is the architectural
  fix for the 10-session plateau seen in Exp 748 and 761.
- **Concrete experiment:** Exp 970 (Milestone 2026.04.75) — PPSEBM Tier 2 Cross-Session Memory:
  implement progressive parameter selection in EmbeddingConstraintStore. Each new dominant
  constraint category (detected by TF-IDF cluster shift) gets a fresh parameter group. EBM
  generates replay pseudo-samples from prior parameter groups. Run 10-session relay. Target:
  non-zero new templates added in sessions 3-10 (vs zero in Exps 748/761).
- **When to incorporate:** Milestone 2026.04.75 (mandatory self-learning experiment per research-program.md).

### ALERT: Zero-Shot Jailbreak Detection via Internal Discrepancy Amplification
- **Paper:** arXiv 2601.03600 (January 2026)
- **What:** Zero-shot jailbreak detector using two complementary classifiers on amplified latent
  representations. No labeled jailbreak examples needed at training time: the method amplifies
  the discrepancy between a model's internal "safe" and "unsafe" representations via contrastive
  perturbation, then trains two independent classifiers on these amplified signals. Achieves
  competitive detection on standard jailbreak benchmarks without any attack-specific training.
- **Relevance to Carnot:** Carnot's Product Roadmap Tier B includes "Safety/Jailbreak Classifier"
  (research-program.md: 2-3 experiments). ALERT's zero-shot approach maps directly onto
  Carnot's energy architecture: the "discrepancy between internal representations" IS an energy
  gap — low discrepancy energy = safe output, high discrepancy energy = jailbreak attempt.
  Key advantage: ALERT's training-free property means Carnot can deploy a jailbreak classifier
  using only the existing Ising/KAN stack, without needing labeled jailbreak training data.
  The two complementary classifiers align with Carnot's cascade (Tier 0 fast + Tier 3 slow).
  This is the foundation for the Tier B safety product, distilled into a small KAN model.
- **Concrete experiment:** Future milestone (2026.04.76+) — Safety/Jailbreak KAN Classifier:
  implement ALERT's discrepancy amplification using Carnot's EmbeddingConstraintStore latent
  space. Train a KAN energy function on the amplified discrepancy signal. Target: AUC > 0.85
  on public jailbreak benchmarks (AdvBench, JailbreakBench) without labeled training data.
- **When to incorporate:** Milestone 2026.04.76 (after ThreeTierPipeline is fully deployed in .75).

## 2026-04-27 arxiv Scan (Milestone 2026.04.77 Planning)

### HalluSAE: Detecting Hallucinations via Sparse Auto-Encoders and Phase-Transition Energy Landscapes
- **Paper:** arXiv 2604.16430 (April 2026)
- **What:** Phase-transition-inspired framework that models hallucination detection as a critical
  shift in latent dynamics. Uses potential energy landscapes derived from Sparse Autoencoder (SAE)
  features to identify hallucination-prone activations through three stages: Potential Energy
  Phase Zone Localization, Hallucination Feature Attribution, and Causal Detection. The potential
  energy landscape identifies layer-wise regions where the model's internal state is near a phase
  boundary — a precursor to hallucination.
- **Relevance to Carnot:** This is a direct energy-based interpretation of the SAE hallucination
  signal that Carnot attempted in Exps 863/878 (HalluSAEGeometricProbe, retired at AUC=0.45).
  The phase-transition framing may explain the prior failure: Carnot used geometric (distance-based)
  energy, but the paper uses potential energy derived from SAE feature activation patterns —
  different signal entirely. The SAE phase-zone localization is a new Tier 0 probe candidate
  that does not repeat the retired HalluSAE geometry approach.
- **Concrete experiment:** Milestone 2026.04.78+ — HalluSAE Phase-Energy Tier 0g: extract SAE
  features from the live Gemma4 model using eleutherai/sae-lens, compute potential energy via
  feature activation patterns (not geometry), compare AUROC vs existing Tier 0 probes. Target
  AUC > 0.65 (retired approach floor). This is NOT a rerun of retired Exps 863/878 — different
  signal source (SAE feature energy vs geometric probe).
- **Prior failure:** Exps 863/878 (HalluSAEGeometricProbe) retired with below_v1 verdict. Root
  cause: geometry-based energy (Euclidean distance) is insensitive to SAE feature activation
  patterns. New approach uses potential energy from feature activation — structurally different.
- **When to incorporate:** Milestone 2026.04.78 (after SAE infrastructure from Tier 0 probes
  is established in .77).

### DiffuTruth: Non-Equilibrium Thermodynamics for LLM Hallucination Detection
- **Paper:** arXiv 2602.11364 (February 2026)
- **What:** DiffuTruth reconceptualizes fact verification via non-equilibrium thermodynamics.
  Factual truths are modeled as stable attractors on a generative manifold; hallucinations are
  unstable regions. The key metric is a "Semantic Energy" computed from diffusion model
  likelihoods — claims with low semantic energy are stable (factual), claims with high semantic
  energy are unstable (hallucinated). Training-free: uses a pre-trained diffusion LM without
  fine-tuning. Reports AUROC 0.70+ on fact verification benchmarks.
- **Relevance to Carnot:** DiffuTruth's Semantic Energy metric is structurally identical to
  Carnot's constraint energy: low energy = constraint satisfied (claim is factual), high energy
  = constraint violated (claim is hallucinated). The key novelty is using a diffusion model's
  generative energy as the verification oracle rather than a discriminative classifier. This
  is relevant to Carnot's Phase 3 vision: an EBM that directly evaluates factual consistency.
  More immediately: DiffuTruth's training-free property means it can be integrated as a
  Tier 0 probe using a locally-hosted diffusion LM without any additional labeled data.
- **Concrete experiment:** Milestone 2026.04.78+ — DiffuTruth Semantic Energy Tier 0h: integrate
  DiffuTruth's likelihood-based energy computation using a local diffusion LM (e.g., from
  HuggingFace Diffusers). Compare AUROC vs Tier 0b (NUP Probe, AUC=1.0 on synthetic) on the
  FOVER live corpus (57 pairs). Target: AUROC improvement as an additive signal to the cascade.
- **When to incorporate:** Milestone 2026.04.78+ (requires diffusion LM infrastructure).

### Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy
- **Paper:** arXiv 2603.06875 (March 2026)
- **What:** Proves that attention computation is exactly one gradient descent step on a classical
  Modern Hopfield energy function. Derives stochastic attention as Langevin sampling on this
  energy: temperature parameter controls the tradeoff between exact memory retrieval (low
  temperature → deterministic attention) and exploratory generation (high temperature →
  stochastic attention). Provides a unified energy-based framework for both attention and MCMC.
- **Relevance to Carnot:** This is the missing theoretical bridge between Carnot's MCMC-based
  Ising/KAN energy evaluation and transformer attention mechanisms. The key insight: if
  attention IS an energy function evaluation, then Carnot can directly replace standard
  attention with a constrained-energy attention — one that minimizes constraint energy during
  the forward pass rather than just post-hoc. This opens a path to Phase 3 guided decoding:
  instead of checking constraints after generation, use the constraint energy as the attention
  energy during generation. Langevin temperature parameter maps to Carnot's beta (inverse temp).
- **Concrete experiment:** Milestone 2026.04.79+ — Energy-Attention Bridge: wire Carnot's IsingEBM
  energy into the Hopfield attention energy formulation. Compare constraint satisfaction rate
  (% responses with zero violations) for standard attention vs energy-guided attention on 50
  GSM8K questions using Gemma4-E4B-it. This is the first experiment toward Phase 3 guided
  decoding that doesn't require a separate post-hoc repair step.
- **When to incorporate:** Milestone 2026.04.79+ (Phase 3 research track, after live pipeline
  baseline confirmed in .77).

### Annealed Langevin Monte Carlo for Multimodal Energy Landscapes
- **Paper:** arXiv 2604.20052 (April 2026)
- **What:** ALMC-ODE provides a principled alternative to standard MCMC for high-dimensional
  multimodal distributions where energy barriers prevent gradient-based samplers from escaping
  local minima. Combines annealed importance sampling with Langevin dynamics via an ODE
  formulation, achieving reliable mixing across energy barriers. Demonstrated on Bayesian
  inference problems with isolated modes.
- **Relevance to Carnot:** Carnot's Langevin SB sampler (Exp 983, LSB, now the default) uses
  parallelized Langevin dynamics but does not explicitly handle multimodal energy landscapes.
  For constraint problems with multiple valid configurations (e.g., GSM8K problems with
  multiple valid intermediate calculation paths), the sampler may get stuck in a local minimum.
  ALMC-ODE's annealing schedule directly addresses this: gradually raising the temperature
  allows the sampler to escape false minima while lowering it converges to the true constraint-
  satisfying configuration. Relevant to KV260 RTL: the annealing schedule is hardware-native
  (simple counter-driven temperature decay).
- **Concrete experiment:** Milestone 2026.04.78+ — ALMC-ODE Sampler: add annealed temperature
  schedule to the LSB sampler (Exp 983 default). Compare convergence on multi-modal constraint
  problems (arithmetic problems with multiple valid intermediate steps). Target: AUC improvement
  on the 57-pair FoVer corpus where multiple reasoning paths are equally valid.
- **When to incorporate:** Milestone 2026.04.78 (after LSB sampler is validated in production
  from .76/.77).

## 2026-04-28 arxiv Scan (Milestone 2026.04.78 Planning)

### Process Reward Models That Think (ThinkPRM)
- **Paper:** arXiv 2504.16828 (April 2026)
- **Authors:** Lightman et al., MIT / Harvard
- **What:** Trains data-efficient process reward models (PRMs) that verify intermediate
  reasoning steps by generating explicit verification chain-of-thought (CoT). Rather than
  requiring expensive step-level discriminative supervision (PRM800K), ThinkPRM leverages
  long-CoT models to reason about whether each step is correct. Achieves 1% of PRM800K's
  training data requirement while outperforming discriminative PRMs on ProcessBench, MATH-500,
  and AIME '24. Out-of-domain gains: +8% on GPQA-Diamond, +4.5% on LiveCodeBench vs
  LLM-as-a-Judge at equal compute budget.
- **Relevance to Carnot:** Carnot's ThinkProbe (Exp 444, Tier 0a) already implements a
  3-step CoT pre-filter that asks a small model to reason about whether a response is
  correct before Ising verification. ThinkPRM validates and extends this: the "generation
  chain-of-thought" verification pattern is equivalent to Carnot's CarnotThinkProbe but
  with a trained verifier instead of a prompted one. Concrete implications: (1) Carnot should
  train ThinkPRM-style on the 57-pair FoVer corpus to produce a trained step-verifier rather
  than relying on zero-shot prompting. (2) ThinkPRM's 8% GPQA-Diamond gain on out-of-domain
  evaluation is the type of generalization Carnot needs: our 57-pair FoVer corpus is in-domain
  math; we want a verifier that generalizes to code, logic, planning. (3) The +4.5% on
  LiveCodeBench shows ThinkPRM adds value ON TOP of strong base models — exactly the
  Carnot-adds-value-on-top-of-SOTA claim we need evidence for.
- **Concrete experiment:** Milestone 2026.04.78 — ThinkPRM-Verified CoT Step Scorer: train a
  small Gemma4/Qwen3.6-based step verifier on FoVer pairs using ThinkPRM's CoT-generation
  approach. Compare to CarnotThinkProbe zero-shot baseline on the 57-pair corpus. Metric: AUC
  for step-level violation detection. Target: ThinkPRM-trained probe beats zero-shot probe.
- **When to incorporate:** Milestone 2026.04.78 (Phase 3 — new research).

### Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning
- **Paper:** arXiv 2601.17223 (January 2026)
- **Authors:** Anon submission
- **What:** Introduces Verifiable Process Reward Models (VPRMs) — a reinforcement learning
  framework where intermediate reasoning steps are verified by deterministic, rule-based
  verifiers rather than neural judges. Applied to medical evidence synthesis where
  guideline-defined criteria enable programmatic step verification. Achieves +20% F1 vs
  SOTA, +6.5% improvement vs outcome-only verification. Key contribution: shows that
  interpretable, auditable rule verification outperforms neural reward modeling when formal
  rules can be expressed.
- **Relevance to Carnot:** Carnot's existing VPRM arithmetic verifier (Exp 454, F1=1.0 vs
  baseline=0.0) implements exactly this principle for arithmetic steps. VPRMs (arXiv 2601.17223)
  validates the approach on a richer domain (medical reasoning) and provides a general
  framework for extending to logic, code, and planning. The "rule-based verifier" is the
  formal analogue of Carnot's Z3-backed VeriCoTStepValidator (Exp 453) — the paper
  provides the formal RL integration missing from Carnot's current pipeline. The +6.5% vs
  outcome-only verification is exactly what Carnot should be measuring: does step-level
  constraint verification beat outcome-only (whole-response) checking?
- **Concrete experiment:** Milestone 2026.04.78 — VPRM Rule-Based Step Verifier: extend
  Exp 454's VPRMArithmeticVerifier to 6+ rule families (arithmetic, comparison, unit
  consistency, logical entailment, code correctness, factual grounding). Evaluate on the
  57-pair FoVer corpus with per-step labels. Target: F1 >= 0.90 on step-level verification.
- **When to incorporate:** Milestone 2026.04.78 (Phase 3 — extends proven Exp 454 pattern).

## 2026-04-28 arxiv Scan (Milestone 2026.04.80 Planning)

### QuantKAN: A Unified Quantization Framework for Kolmogorov-Arnold Networks
- **Paper:** arXiv 2511.18689 (November 2025)
- **Authors:** Zhang et al.
- **What:** Proposes post-training quantization (PTQ) and quantization-aware training (QAT)
  for KANs. Key finding: B-spline activations in KANs are highly sensitive to weight precision
  due to the basis-function product structure — naive INT8 quantization loses 15–30% accuracy,
  but the QuantKAN framework recovers to within 1% of FP32 via per-knot scale factors and
  activation-range-aware rounding. Includes FPGA-deployable fixed-point recipes for KAN layers.
- **Relevance to Carnot:** Carnot's GS-KAN energy tier (KANELÉ-inspired, G=4 shared basis)
  needs to run on KV260 (constrained BRAM). QuantKAN's FPGA recipes directly address the
  LUT/BRAM budget constraint identified in Exp 1019 (GS-KAN FPGA analysis): INT8 KAN activations
  cut DSP usage by ~4x vs FP32. The per-knot scale factor approach is compatible with the
  Multilevel KAN training (arXiv 2603.04827) — quantize after multilevel convergence.
- **Concrete experiment:** Milestone 2026.04.80 — apply QuantKAN INT8 recipe to GS-KAN energy
  tier before FPGA synthesis. Target: verify BRAM < 512 blocks and DSP48 < 200 on KV260.
- **When to incorporate:** Milestone 2026.04.80 (Phase 2 — FPGA synthesis prerequisite for KV260 first light).

### MetaQA: Hallucination Detection in LLMs via Metamorphic Relations
- **Paper:** arXiv 2502.15844 (February 2026)
- **Authors:** Anon submission (OpenReview)
- **What:** Frames hallucination detection as a metamorphic testing problem: a claim C is a
  hallucination if semantically equivalent rephrasing C' yields inconsistent model outputs.
  Introduces a suite of 12 metamorphic relations (negation, synonym substitution, entity
  replacement, etc.) that a non-hallucinating model must satisfy. Achieves 88% detection rate
  on TruthfulQA with 0 training labels — a purely black-box, model-agnostic test.
- **Relevance to Carnot:** Carnot's Tier 0 verification chain (SpilledEnergy, NUP Probe,
  HallucinationBasin) currently requires labeled FoVer data to train probes. MetaQA's
  zero-shot metamorphic approach could generate additional training signal for the FoVer
  corpus WITHOUT needing Z3 labels: if a CoT step fails metamorphic consistency, it is
  labeled as a potential hallucination. This directly addresses the FoVer corpus expansion
  bottleneck (57 → 500+ pairs).
- **Concrete experiment:** Milestone 2026.04.80 — integrate MetaQA's top-5 metamorphic
  relations into the FoVer corpus expansion pipeline as a weak labeler alongside Z3. Target:
  50+ additional labeled pairs from metamorphic inconsistency detection.
- **When to incorporate:** Milestone 2026.04.80 (Phase 1 — directly addresses FoVer bottleneck).

### Ontology Neural Networks for Topologically Conditioned Constraint Satisfaction
- **Paper:** arXiv 2601.05304 (January 2026)
- **Authors:** Anon submission
- **What:** Proposes a neural architecture where constraint satisfaction is encoded as
  topological invariants over a learned ontology graph. The ontology is a DAG of constraint
  classes; the network learns to route activations through constraint-satisfied paths.
  Achieves 95% constraint satisfaction on combinatorial planning benchmarks vs 73% for
  unconstrained baseline. Key: constraints are type-level, not loss-level — violations are
  structurally impossible in the architecture, not just penalized.
- **Relevance to Carnot:** Carnot's SOS-Integrated KAN (research-studying.md) and the
  Phase 3 EBT foundation model both need constraint satisfaction as a structural property,
  not a training objective. Ontology NN's topological routing is the neural analogue of
  SOS-on-derivative for monotonicity — and it generalizes to arbitrary constraint classes
  (arithmetic consistency, logical entailment, factual grounding). Relevant to Phase 3
  non-autoregressive reasoning path where constraint satisfaction must be verified at
  generation time without Z3 calls.
- **Concrete experiment:** Milestone 2026.04.81+ — after VPRM rule families are established
  in .80, evaluate whether Ontology NN routing could replace the Z3-backed VeriCoTStepValidator
  for constraint classes where Z3 is too slow (>1s per step).
- **When to incorporate:** Milestone 2026.04.81 (Phase 3 — requires VPRM baseline from .80 first).

## 2026-04-29 arxiv Scan (Milestone 2026.04.81 Planning)

### Self-Distilled RLVR (arXiv 2604.03128)
- **Paper:** arXiv 2604.03128 (April 2026)
- **What:** Combines RL with verifiable rewards (RLVR) and self-distillation. Key finding: pure
  privileged-teacher signals cause information leakage and unstable long-horizon training; the
  self-distillation component provides token-level policy adjustment stabilizing convergence.
  Without self-distillation, RLVR collapses on multi-step reasoning tasks.
- **Relevance to Carnot:** Directly validates Carnot's energy-verifier design. The instability of
  pure teacher distillation is exactly what Zenil's Theorem 4 predicts — the self-distillation
  loop needs an exogenous grounding signal (α_t > 0). Carnot's energy function IS that stable,
  non-leaking ground truth. Citable in Phase 3 position paper.
- **Concrete experiment:** Milestone 2026.04.81 (Exp 1046) — cite as supporting evidence for the
  α_t grounding measurement module.
- **When to incorporate:** .81 (Exp 1046 Zenil FR-11).

### Reinforcement Learning via Self-Distillation (SDPO, arXiv 2601.20802)
- **Paper:** arXiv 2601.20802 (January 2026)
- **What:** Self-Distillation Policy Optimization re-uses the model conditioned on rich textual
  feedback as a self-teacher. Outperforms scalar-reward RL on scientific reasoning, tool use,
  and competitive programming. The feedback-conditioned model produces qualitatively richer
  supervision than a reward scalar alone.
- **Relevance to Carnot:** SDPO's feedback-conditioned self-teacher is the mechanistic
  implementation of the Zenil α_t grounding term. The verifier's CoT explanation of *why* an
  output is high-energy is the textual feedback. Combining SDPO's policy update with Carnot's
  energy function as the reward signal is a concrete Phase 3 experiment.
- **Concrete experiment:** Milestone .82+ — implement SDPO-style policy update using Carnot
  energy + violation explanation as the feedback signal. Compare to temperature-only SSD baseline.
- **When to incorporate:** .82 (after Zenil α_t module is deployed in .81).

### KAN-SAs: Efficient Acceleration of KANs on Systolic Arrays (arXiv 2512.00055)
- **Paper:** arXiv 2512.00055 (December 2025)
- **What:** Non-recursive B-spline computation plus sparsity-aware mapping achieves 100% systolic
  array utilization and 50% reduction in clock cycles vs equivalent-area conventional systolic
  arrays. The key insight: B-spline recursion can be unrolled statically, eliminating data
  hazards that kill utilization on standard SAs.
- **Relevance to Carnot:** The KAN tier (carnot-kan) is a deployment target for the KV260 FPGA.
  KAN-SAs' non-recursive mapping directly addresses the reason standard FPGA KAN implementations
  are inefficient — recursion causes pipeline bubbles. Complementary to KANELÉ (arXiv 2512.12850,
  already in this file) which focuses on LUT-based evaluation rather than systolic mapping.
- **Concrete experiment:** Milestone .82 — after KV260 first light (.81), implement KAN-SA style
  non-recursive B-spline unrolling in the KAEMEnergy Verilog synthesis target.
- **When to incorporate:** .82 FPGA synthesis track.

### GenCP: Large Language Model Meets Constraint Propagation (arXiv 2505.24012)
- **Paper:** arXiv 2505.24012 (May 2026)
- **What:** Formulates LLM text generation as a Constraint Satisfaction Problem. Uses masked LMs
  for bidirectional constraint propagation; domain preview via MLM calls significantly improves
  feasible solution rate on COLLIE benchmarks. The bidirectionality is the key advance —
  existing constrained generation methods are strictly left-to-right.
- **Relevance to Carnot:** GenCP's bidirectional constraint propagation is the neural analog of
  Carnot's repair pass. Currently Carnot verifies POST-generation and repairs via re-sampling.
  GenCP integrates constraint enforcement INTO generation. Combining GenCP's MLM-based propagation
  with Carnot's energy function as the constraint oracle could replace heuristic repair with
  constraint-sound generation — a major Phase 1 improvement.
- **Concrete experiment:** Milestone .82+ — prototype GenCP-style bidirectional propagation for
  arithmetic constraint satisfaction. Use Carnot's energy function as the constraint oracle
  replacing GenCP's hard-coded constraint checks.
- **When to incorporate:** .82 (research path to guided decoding).

### PRISM: PRM-Guided Inference Scaling (arXiv 2603.02479)
- **Paper:** arXiv 2603.02479 (March 2026)
- **What:** PRM-guided refinement and aggregation of solution populations achieves 90.0% on
  AIME25 and 75.4% on HMMT25, matching or exceeding larger models. Net-directional correction
  holds even when initial populations contain few correct solutions — the PRM can rescue
  bad populations via guided resampling.
- **Relevance to Carnot:** PRISM's step-level PRM guidance is the inference-time complement to
  Carnot's training-time verification. Using Carnot's energy verifier as the PRM in PRISM-style
  search gives a principled test-time scaling path. PRISM's finding that PRM guidance works even
  on bad populations means Carnot's verifier could improve outputs from weaker models, directly
  addressing the precision ceiling problem (Exp 184: 0% net improvement on 3B models).
- **Concrete experiment:** Milestone .82+ — implement PRISM-style resampling guided by Carnot's
  energy function. Run on GSM8K with Qwen3.6-35B-A3B. Compare to temperature-only baseline.
- **When to incorporate:** .82 (after Triple Integration validates the cascade in .81).

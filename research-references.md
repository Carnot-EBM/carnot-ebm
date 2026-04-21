# Research References & Future Considerations

Items filed here are technologies, papers, repos, and ideas to consider
in future research milestones. The research conductor and planning agent
should read this file when designing new milestones.

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

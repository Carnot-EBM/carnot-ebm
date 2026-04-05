# arXiv Literature Scan -- 2026-04-05

Scan of recent (2025--2026) arXiv papers relevant to the Carnot EBM framework.
Topics: energy-based transformers, JEPA, EBM for code/verification, self-supervised energy training, thermodynamic computing.

---

## 1. Energy-Based Transformers

### 1.1 Energy-Based Transformers are Scalable Learners and Thinkers
- **arXiv:** [2507.02092](https://arxiv.org/abs/2507.02092)
- **Summary:** Introduces EBT, a Transformer designed as an EBM that assigns energy to input/prediction pairs; achieves superior pretraining scaling across parameters, depth, data, and FLOPs via gradient-descent energy minimization at inference.
- **Actionable for Carnot:** This is the most directly relevant paper. EBT's decoder-only (GPT-style) and bidirectional (BERT/DiT-style) variants validate Carnot's core thesis that EBMs can scale. Their energy-scalar uncertainty metric maps directly to Carnot's rejection-sampling experiments. Consider implementing EBT attention patterns in `carnot-boltzmann`.

### 1.2 NRGPT: An Energy-based Alternative for GPT
- **arXiv:** [2512.16762](https://arxiv.org/abs/2512.16762) (Accepted ICLR 2026)
- **Summary:** Minimally modifies GPT to unify it with the EBM framework; inference becomes gradient descent on an energy landscape over token embeddings, with empirical evidence of improved overfitting resistance.
- **Actionable for Carnot:** NRGPT's "minimal modification" approach is a concrete recipe for Carnot's Python Boltzmann tier. Their Shakespeare/ListOPS/OpenWebText benchmarks provide replicable evaluation targets. Their overfitting-resistance claim is worth validating with Carnot's Ising tier on small datasets.

### 1.3 Energy Transformer (ET)
- **arXiv:** [2302.07253](https://arxiv.org/abs/2302.07253)
- **Summary:** Attention layers redesigned to minimize a specifically engineered energy function over token relationships; earlier foundational work that EBT builds upon.
- **Actionable for Carnot:** Reference architecture for Carnot's attention-as-energy-minimization design pattern.

---

## 2. EBM + Language Model Theory

### 2.1 Autoregressive LMs are Secretly Energy-Based Models
- **arXiv:** [2512.15605](https://arxiv.org/abs/2512.15605)
- **Summary:** Establishes an explicit bijection between autoregressive models and EBMs in function space; provides theoretical error bounds for distilling EBMs into ARMs and explains lookahead/planning capabilities of next-token prediction.
- **Actionable for Carnot:** Foundational theory for Carnot's autoresearch loop. If ARMs are secretly EBMs, then Carnot's energy functions can be used to analyze and improve existing LLM behavior. The distillation bounds are directly useful for the Rust transpilation pipeline (JAX EBM -> Rust ARM).

### 2.2 A Theoretical Lens for RL-Tuned LMs via EBMs
- **arXiv:** [2512.18730](https://arxiv.org/abs/2512.18730)
- **Summary:** Proves that the optimal KL-regularized RLHF policy has closed-form EBM structure; derives monotonic convergence, bounded hitting times, and spectral-gap mixing for instruction-tuned models.
- **Actionable for Carnot:** Provides the theoretical grounding for using Carnot's energy functions as reward models in RLHF. The spectral-gap analysis could inform Carnot's MCMC sampling convergence guarantees.

### 2.3 Matching Features, Not Tokens: Energy-Based Fine-Tuning of LMs
- **arXiv:** [2603.12248](https://arxiv.org/abs/2603.12248) (March 2026)
- **Summary:** Introduces EBFT -- feature-matching fine-tuning that targets sequence-level statistics without a task-specific verifier; uses strided block-parallel sampling for efficient on-policy gradient updates.
- **Actionable for Carnot:** EBFT's feature-matching objective is a direct alternative to Carnot's current logprob-based rejection sampling. Their block-parallel sampling could accelerate Carnot's JAX training pipeline. Very recent (March 2026) -- worth close tracking.

### 2.4 Energy-Based Reward Models for Robust LM Alignment
- **arXiv:** [2504.13134](https://arxiv.org/abs/2504.13134) (Accepted COLM 2025)
- **Summary:** EBRM is a lightweight post-hoc refinement framework that enhances reward model robustness via conflict-aware data filtering and label-noise-aware contrastive training; up to 5.97% improvement on safety-critical alignment.
- **Actionable for Carnot:** EBRM's noise-aware contrastive training is relevant to Carnot's composite energy experiments (logprob + structural). Their post-hoc refinement pattern could be adapted as a Carnot energy layer on top of existing reward models.

---

## 3. Energy-Based Diffusion for Language

### 3.1 Energy-Based Diffusion Language Models for Text Generation (EDLM)
- **arXiv:** [2410.21357](https://arxiv.org/abs/2410.21357) (ICLR 2025)
- **Summary:** Combines EBMs with discrete diffusion for full-sequence-level denoising; captures inter-token correlations that factorized diffusion models miss, reducing accumulated error and improving sampling efficiency.
- **Actionable for Carnot:** EDLM's sequence-level energy function is architecturally aligned with Carnot's Boltzmann tier. Their approach to reducing sampling steps while maintaining quality maps to Carnot's efficiency goals. Consider EDLM-style diffusion as a fourth sampling strategy alongside greedy/rejection/MCMC.

---

## 4. JEPA and Joint Embedding Architectures

### 4.1 V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- **arXiv:** [2506.09985](https://arxiv.org/abs/2506.09985) (June 2025)
- **Summary:** Pre-trained on 1M+ hours of video; achieves SOTA on action anticipation and can be post-trained as a latent action-conditioned world model for zero-shot robotic planning.
- **Actionable for Carnot:** V-JEPA 2's world-model post-training validates the JEPA paradigm at scale. Their latent-space planning approach could inform Carnot's autoresearch loop -- using energy landscapes for planning over code transformations rather than physical actions.

### 4.2 VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language
- **arXiv:** [2512.10942](https://arxiv.org/abs/2512.10942)
- **Summary:** Predicts continuous embeddings of target text rather than autoregressive tokens; achieves stronger VLM performance with 50% fewer trainable parameters.
- **Actionable for Carnot:** VL-JEPA's continuous-embedding prediction (vs. discrete token prediction) is the JEPA analog of what Carnot aims to do with energy-based code representations. Their parameter efficiency gains suggest that Carnot's Ising tier could be surprisingly competitive.

### 4.3 ACT-JEPA: Joint-Embedding for Efficient Policy Representation Learning
- **arXiv:** [2501.14622](https://arxiv.org/abs/2501.14622) (January 2025, updated March 2026)
- **Summary:** Unifies imitation learning and self-supervised learning by jointly predicting action sequences and latent observation sequences end-to-end.
- **Actionable for Carnot:** ACT-JEPA's joint action/observation prediction is a template for Carnot's autoresearch agent: the "action" is a code edit, the "observation" is the resulting test/energy landscape.

### 4.4 Var-JEPA: A Variational Formulation of JEPA
- **arXiv:** [2603.20111](https://arxiv.org/abs/2603.20111) (March 2026)
- **Summary:** Derives a generative JEPA from variational inference with a corresponding ELBO objective that naturally prevents representational collapse.
- **Actionable for Carnot:** Var-JEPA's collapse-prevention via ELBO is directly relevant to Carnot's training stability. The variational formulation bridges JEPA and traditional EBM training, which is exactly Carnot's theoretical gap.

### 4.5 Audio-JEPA
- **arXiv:** [2507.02915](https://arxiv.org/abs/2507.02915) (June 2025)
- **Summary:** Applies JEPA to audio spectrograms using ViT backbone with masked-patch prediction in latent space.
- **Actionable for Carnot:** Demonstrates JEPA generality across modalities; the masked-patch prediction in latent space pattern could be adapted for code-token masking in Carnot's training.

---

## 5. Self-Supervised Energy Training

### 5.1 Scalable EBMs via Adversarial Training: Unifying Discrimination and Generation
- **arXiv:** [2510.13872](https://arxiv.org/abs/2510.13872) (October 2025, updated March 2026)
- **Summary:** Replaces unstable SGLD-based training with adversarial PGD-contrastive approach; first EBM-based hybrid to scale to ImageNet 256x256 with high stability, matching ARMs and beating diffusion models on generation quality.
- **Actionable for Carnot:** Their SGLD replacement with adversarial training is directly applicable to Carnot's training stability issues. The BCE-based energy optimization could replace or augment Carnot's current contrastive divergence. High priority -- addresses Carnot's known scaling challenges.

### 5.2 Understanding Adversarial Training with Energy-based Models
- **arXiv:** [2505.22486](https://arxiv.org/abs/2505.22486) (May 2025)
- **Summary:** Theoretical analysis of adversarial training through the EBM lens, providing understanding of when and why adversarial EBM training succeeds.
- **Actionable for Carnot:** Companion theory paper to 2510.13872; useful for understanding the conditions under which Carnot's energy training will be stable.

### 5.3 Contrastive Self-Supervised Learning at the Edge: An Energy Perspective
- **arXiv:** [2510.08374](https://arxiv.org/abs/2510.08374) (October 2025)
- **Summary:** Benchmarks SimCLR, MoCo, SimSiam, and Barlow Twins for energy consumption on edge devices; SimCLR has lowest energy cost.
- **Actionable for Carnot:** Practical guidance for Carnot's Ising tier (smallest model) deployment. SimCLR-style contrastive training may be the right default for resource-constrained inference.

---

## 6. Thermodynamic Computing / Analog EBM Hardware

### 6.1 Generative Thermodynamic Computing
- **arXiv:** [2506.15121](https://arxiv.org/abs/2506.15121) (June 2025)
- **Summary:** Shows thermodynamic computers resemble nonequilibrium continuous-spin Boltzmann machines; proposes analog hardware (mechanical, electrical, superconducting oscillators) where computation is encoded in the energy landscape.
- **Actionable for Carnot:** This is the hardware roadmap for Carnot's Boltzmann tier. If Carnot's energy functions can be mapped to continuous-spin Hamiltonians, they become candidates for analog acceleration. The Rust core's numerical precision will matter for hardware mapping fidelity.

### 6.2 Training Thermodynamic Computers by Gradient Descent
- **arXiv:** [2509.15324](https://arxiv.org/abs/2509.15324) (September 2025)
- **Summary:** Shows thermodynamic computers can solve linear algebra (matrix inversion) in equilibrium using the Boltzmann distribution, with gradient-descent training of the energy landscape.
- **Actionable for Carnot:** The gradient-descent training of physical energy landscapes is exactly what Carnot simulates in software. This paper validates that Carnot's `carnot-boltzmann` crate could eventually compile to hardware.

### 6.3 Digitally Optimized Initializations for Fast Thermodynamic Computing
- **arXiv:** [2603.24183](https://arxiv.org/abs/2603.24183) (March 2026)
- **Summary:** Hybrid digital-thermodynamic algorithm that uses classical preprocessing to suppress slow relaxation modes (inspired by the Mpemba effect), accelerating thermalization for matrix inversion and determinant computation.
- **Actionable for Carnot:** The hybrid digital-analog pattern is directly relevant to Carnot's architecture: use Rust for digital initialization, then let the energy landscape relax. Their Mpemba-inspired initialization could improve MCMC warm-starting in Carnot's sampling.

### 6.4 Nonlinear Thermodynamic Computing Out of Equilibrium
- **arXiv:** [2412.17183](https://arxiv.org/abs/2412.17183) (December 2024, updated 2025)
- **Summary:** Extends thermodynamic computing beyond linear equilibrium to nonlinear out-of-equilibrium Langevin dynamics, enabling nonlinear computations at specified times.
- **Actionable for Carnot:** Nonlinear out-of-equilibrium dynamics are needed for Carnot's more complex energy landscapes. The Langevin computer formalism could inform Carnot's MCMC sampler design.

### 6.5 An Efficient Probabilistic Hardware Architecture for Diffusion-like Models
- **arXiv:** [2510.23972](https://arxiv.org/abs/2510.23972) (October 2025)
- **Summary:** Proposes specialized stochastic circuitry for sampling from Boltzmann distributions, exploiting sparsity and locality constraints of physical EBM implementations.
- **Actionable for Carnot:** Hardware-aware sparsity constraints should influence Carnot's model design. If the Ising tier enforces locality and sparsity, it becomes a candidate for this class of hardware accelerators.

---

## Priority Ranking for Carnot Integration

| Priority | Paper | Why |
|----------|-------|-----|
| **P0** | EBT (2507.02092) | Validates EBM-transformer scaling; direct architecture reference |
| **P0** | Scalable EBMs via Adversarial (2510.13872) | Solves training stability; replaces SGLD |
| **P0** | ARMs are secretly EBMs (2512.15605) | Foundational theory for ARM<->EBM bridge |
| **P1** | NRGPT (2512.16762) | Concrete GPT-to-EBM recipe, ICLR 2026 |
| **P1** | EDLM (2410.21357) | Sequence-level energy for diffusion, ICLR 2025 |
| **P1** | EBFT (2603.12248) | Feature-matching fine-tuning alternative |
| **P1** | Var-JEPA (2603.20111) | Collapse prevention via ELBO, bridges JEPA and EBM |
| **P2** | V-JEPA 2 (2506.09985) | World-model planning validates JEPA at scale |
| **P2** | Generative Thermo Computing (2506.15121) | Hardware roadmap for Boltzmann tier |
| **P2** | Digital-Thermo Init (2603.24183) | Hybrid digital-analog warm-starting |
| **P2** | EBRM (2504.13134) | Energy-based reward models for alignment |
| **P3** | RL-Tuned LMs via EBMs (2512.18730) | RLHF theory grounding |
| **P3** | ACT-JEPA (2501.14622) | Autoresearch agent template |
| **P3** | Training Thermo by GD (2509.15324) | Validates software->hardware path |

---

## Suggested Next Steps

1. **Read P0 papers in full** -- especially EBT and adversarial-training EBM for immediate architectural decisions
2. **Prototype NRGPT-style minimal modification** on Carnot's Ising tier as a quick validation
3. **Evaluate adversarial training (2510.13872)** as replacement for contrastive divergence in Carnot's training loop
4. **Track Var-JEPA** -- its ELBO-based collapse prevention may solve Carnot's representational stability issues
5. **Map Carnot's Boltzmann Hamiltonians** to the continuous-spin format from generative thermodynamic computing (2506.15121) for future hardware targeting

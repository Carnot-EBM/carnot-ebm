# arXiv Literature Scan -- 2026-04-05

Scan of recent (2025--2026) arXiv papers relevant to the Carnot EBM framework.
Topics: energy-based transformers, JEPA, EBM for code/verification, self-supervised energy training, thermodynamic computing.

---

## 1. Energy-Based Transformers

### 1a. Energy-Based Transformers are Scalable Learners and Thinkers
- **arXiv:** [2507.02092](https://arxiv.org/abs/2507.02092) (Jul 2025)
- **Summary:** EBTs assign energy to input-prediction pairs and optimize via gradient descent at inference; they scale 35% faster than Transformer++ in data/params/FLOPs and gain 29% more from System-2 thinking on language tasks.
- **Actionable for Carnot:** Direct validation of Carnot's thesis that energy-based inference is competitive with autoregressive decoding. The EBT architecture (energy function over candidate predictions, iterative refinement) maps closely to Carnot's Boltzmann tier. Consider implementing their scaling benchmarks as a comparison target, and adopting their "thinking budget" parameterization for Carnot inference.

### 1b. Transformers as Intrinsic Optimizers: Forward Inference through the Energy Principle
- **arXiv:** [2511.00907](https://arxiv.org/abs/2511.00907) (Nov 2025)
- **Summary:** Unifies attention mechanisms under an energy framework with local energy, global energy, and optimization components -- different attention forms arise from different "recipes" within the framework.
- **Actionable for Carnot:** Theoretical grounding for Carnot's energy-attention design. Their taxonomy (local vs. global energy) could inform how Carnot structures multi-scale energy landscapes in the Boltzmann/Gibbs/Ising tiers.

---

## 2. JEPA and Joint Embedding Architectures

### 2a. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures
- **arXiv:** [2509.14252](https://arxiv.org/abs/2509.14252) (Sep 2025, LeCun + Balestriero)
- **Summary:** First JEPA objective for LLMs -- combines reconstruction loss with a JEPA embedding-prediction loss; outperforms standard LLM training on GSM8K, Spider, etc., while being robust to overfitting.
- **Actionable for Carnot:** Directly relevant to Carnot's JEPA integration ambitions. Their dual-objective (reconstruction + JEPA) could be adopted for Carnot's language-tier training. The multi-view learning approach (multiple views of same knowledge) aligns with Carnot's energy-landscape viewpoint.

### 2b. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- **arXiv:** [2506.09985](https://arxiv.org/abs/2506.09985) (Jun 2025, Meta)
- **Summary:** Action-free JEPA pretrained on 1M+ hours of video achieves SOTA on action anticipation and strong motion understanding (77.3% SSv2).
- **Actionable for Carnot:** Demonstrates that JEPA scales to massive data without action labels. The training recipe (masking + latent prediction at scale) could inform Carnot's approach to self-supervised pretraining of energy models on large corpora.

### 2c. VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language
- **arXiv:** [2512.10942](https://arxiv.org/abs/2512.10942) (Dec 2025)
- **Summary:** Predicts continuous text embeddings instead of tokens; matches classical VLMs with 50% fewer trainable parameters and 2.85x fewer decoding operations via selective decoding.
- **Actionable for Carnot:** The continuous-embedding prediction (vs. discrete token prediction) is exactly the paradigm Carnot should pursue. Their selective decoding strategy could reduce inference cost in Carnot's energy-minimization loop.

### 2d. Audio-JEPA: Joint-Embedding Predictive Architecture for Audio
- **arXiv:** [2507.02915](https://arxiv.org/abs/2507.02915) (Jul 2025)
- **Summary:** ViT backbone predicts latent representations of masked spectrogram patches; matches wav2vec 2.0 with <1/5 the training data.
- **Actionable for Carnot:** Data efficiency of JEPA is a key selling point. If Carnot can achieve similar data-efficiency gains over autoregressive baselines, that's a major differentiator.

---

## 3. EBMs for Code Generation and Verification

### 3a. Autoregressive Language Models are Secretly Energy-Based Models
- **arXiv:** [2512.15605](https://arxiv.org/abs/2512.15605) (Dec 2025, Blondel et al.)
- **Summary:** Establishes an explicit bijection between autoregressive models (ARMs) and energy-based models (EBMs) in function space; derives equivalence between supervised learning of ARMs and EBMs; provides theoretical error bounds for EBM-to-ARM distillation.
- **Actionable for Carnot:** Foundational theory. This proves Carnot's core bet is sound -- that EBMs can match ARM capabilities. The bijection and distillation bounds give Carnot a principled path to "compile" energy models into efficient autoregressive form for deployment, or vice versa.

### 3b. A Theoretical Lens for RL-Tuned Language Models via Energy-Based Models
- **arXiv:** [2512.18730](https://arxiv.org/abs/2512.18730) (Dec 2025)
- **Summary:** Shows that KL-regularized RL policies have closed-form EBM structure; proves monotonic KL convergence and exponential mixing for instruction-tuned models.
- **Actionable for Carnot:** Provides theoretical backing for using energy landscapes to guide RL-based fine-tuning. Could inform Carnot's approach to self-improvement loops (the autoresearch pipeline) -- the detailed-balance property suggests energy-landscape navigation is well-behaved.

### 3c. Energy-Based Diffusion Language Models for Text Generation (EDLM)
- **arXiv:** [2410.21357](https://arxiv.org/abs/2410.21357) (Oct 2024, published ICLR 2025)
- **Summary:** Full-sequence energy model at each diffusion step fixes the train/sample mismatch in discrete diffusion models; addresses the independent-token approximation error.
- **Actionable for Carnot:** EDLM's approach to full-sequence energy scoring (vs. per-token) is directly applicable to Carnot's code verification use case. A whole-program energy function could detect inconsistencies that token-level models miss.

---

## 4. Self-Supervised Energy Training

### 4a. Semantic Energy: Detecting LLM Hallucination Beyond Entropy
- **arXiv:** [2508.14496](https://arxiv.org/abs/2508.14496) (Aug 2025)
- **Summary:** Uses Boltzmann-inspired energy distribution on penultimate-layer logits with semantic clustering to detect hallucinations; outperforms semantic entropy.
- **Actionable for Carnot:** Directly validates Carnot's hallucination-detection direction (already prototyped in the conductor). Their Boltzmann energy formulation on logits could be integrated into Carnot's hallucination detector EBM as an additional signal or baseline comparison.

### 4b. Spilled Energy in Large Language Models
- **arXiv:** [2602.18671](https://arxiv.org/abs/2602.18671) (Feb 2026)
- **Summary:** Training-free EBM-based method for hallucination detection that generalizes across tasks; mathematically principled using energy-based model framework.
- **Actionable for Carnot:** A training-free energy-based hallucination detector is complementary to Carnot's trained detector. Could serve as a lightweight baseline or be combined with Carnot's learned energy landscape for ensemble detection.

### 4c. Contrastive Self-Supervised Learning at the Edge: An Energy Perspective
- **arXiv:** [2510.08374](https://arxiv.org/abs/2510.08374) (Oct 2025)
- **Summary:** Evaluates SimCLR, MoCo, SimSiam, Barlow Twins for edge deployment with energy profiling; characterizes compute/energy tradeoffs.
- **Actionable for Carnot:** Relevant to Carnot's Ising (small) tier design. Energy profiling methodology could be adopted to benchmark Carnot's Rust inference against these baselines on edge hardware.

### 4d. Training Energy-Based Models with Diffusion Contrastive Divergences
- **arXiv:** [2307.01668](https://arxiv.org/abs/2307.01668) (2023, but foundational)
- **Summary:** Replaces MCMC sampling in contrastive divergence with diffusion model samples, breaking the computational-burden/validity tradeoff.
- **Actionable for Carnot:** Training methodology. If Carnot adopts diffusion-based sampling for EBM training (replacing expensive MCMC), this paper provides the theoretical justification and practical recipe.

---

## 5. Thermodynamic Computing / Analog EBM Hardware

### 5a. Generative Thermodynamic Computing
- **arXiv:** [2506.15121](https://arxiv.org/abs/2506.15121) (Jun 2025)
- **Summary:** Generative modeling via Langevin dynamics in physical systems (mechanical, electrical, or superconducting oscillator networks); information encoded in the energy landscape rather than neural network weights.
- **Actionable for Carnot:** Long-term hardware target. Carnot's energy landscapes could potentially be "compiled" to thermodynamic hardware parameters. The oscillator-network formulation maps naturally to Carnot's Ising-tier energy functions.

### 5b. Training Thermodynamic Computers by Gradient Descent
- **arXiv:** [2509.15324](https://arxiv.org/abs/2509.15324) (Sep 2025, Whitelam)
- **Summary:** Trains thermodynamic computers via gradient descent using a teacher-student scheme; estimates 7+ orders of magnitude energy advantage over digital implementations.
- **Actionable for Carnot:** The teacher-student training scheme (digital NN teaches thermodynamic hardware) is exactly the deployment path for Carnot models on analog hardware. The 10^7x energy advantage figure motivates investment in hardware-compatible energy function design.

### 5c. Digitally Optimized Initializations for Fast Thermodynamic Computing
- **arXiv:** [2603.24183](https://arxiv.org/abs/2603.24183) (Mar 2026)
- **Summary:** Hybrid digital-thermodynamic algorithm uses optimized initializations (inspired by Mpemba effect) to suppress slow relaxation modes, accelerating thermalization.
- **Actionable for Carnot:** Practical optimization for thermodynamic deployment. Carnot's digital inference could compute smart initializations before handing off to analog hardware, combining digital precision with analog energy efficiency.

### 5d. Thermodynamic Bounds on Energy Use in Quasi-Static Deep Neural Networks
- **arXiv:** [2503.09980](https://arxiv.org/abs/2503.09980) (Mar 2025)
- **Summary:** Even ideal quasi-static analog implementations require finite dissipation scaling with parameter count for training, but analog inference can be vastly more energy-efficient than digital.
- **Actionable for Carnot:** Sets theoretical bounds on what thermodynamic hardware can achieve. Reinforces the strategy of training digitally (Carnot's JAX/Rust pipeline) and deploying to analog hardware for inference only.

### 5e. Thermal Analog Computing: Matrix-Vector Multiplication
- **arXiv:** [2503.22603](https://arxiv.org/abs/2503.22603) (Mar 2025)
- **Summary:** Demonstrates matrix-vector multiplication using inverse-designed metastructures with heat conduction as signal carrier.
- **Actionable for Carnot:** Novel hardware primitive. If matmul can be done thermally, Carnot's linear-algebra-heavy energy computations could map to such hardware. Worth tracking for Carnot's hardware abstraction layer.

---

## Summary of High-Priority Actions for Carnot

| Priority | Action | Source Papers |
|----------|--------|---------------|
| **P0** | Benchmark against EBT scaling results | 2507.02092 |
| **P0** | Integrate LLM-JEPA dual-objective training | 2509.14252 |
| **P0** | Validate hallucination detector against Semantic Energy baseline | 2508.14496, 2602.18671 |
| **P1** | Adopt EDLM full-sequence energy scoring for code verification | 2410.21357 |
| **P1** | Implement ARM<->EBM bijection for model distillation | 2512.15605 |
| **P1** | Study continuous-embedding prediction (VL-JEPA style) for Carnot inference | 2512.10942 |
| **P2** | Design hardware-compatible energy functions for thermodynamic deployment | 2506.15121, 2509.15324 |
| **P2** | Explore diffusion contrastive divergence for EBM training | 2307.01668 |
| **P2** | Energy profiling of Ising tier on edge hardware | 2510.08374 |
| **P3** | Track hybrid digital-analog initialization schemes | 2603.24183 |
| **P3** | Monitor thermal analog computing primitives | 2503.22603 |

---

*Generated 2026-04-05. Next scan recommended: 2026-05-05.*

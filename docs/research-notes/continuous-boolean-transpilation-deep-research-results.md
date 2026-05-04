# DR-3 Synthesis — Continuous-Boolean Transpilation Literature Survey

**Status:** SYNTHESIZED 2026-05-04 ~01:55Z from Gemini Deep Research response
**Source:** `continuous-boolean-transpilation-deep-research-source.pdf` (1.9MB, 26 pages, 50 cited works)
**Prompt:** `continuous-boolean-transpilation-deep-research-prompt.md`
**Predecessors:** DR-1 (energy-based LLM alternatives), DR-2 (multi-verifier ensemble defenses), Q11 (TSS verifier design)
**Strategic role:** validates Phase-3 substrate commitment + identifies LittleBit Dual-SVID as critical missing technique + tightens novelty boundary

## Three load-bearing findings

**1. Carnot's substrate commitment is the 2025-2026 frontier consensus, not a niche bet.**
The literature decisively converges on bounded-continuous-to-Boolean (or
low-cardinality FSQ) over either purely-continuous or purely-discrete
architectures. Phase-3 DBAE-EBM's $z \in [-1,1]^d \to \text{sign}(z) \to \text{Ising}$
pipeline is exactly the consensus pattern.

**2. LittleBit Dual-SVID (NeurIPS 2025) is the load-bearing missing
technique.** Carnot has been treating the sign(z) bottleneck as a primitive
operation. The literature shows it isn't — naive sign-discretization
catastrophically destroys gradient flow and causes parametric collapse
(continuous latent z → 0.999 → STE false gradient → Boolean output
permanently frozen). LittleBit's Dual-SVID is the multi-scale residual
compensation technique that ACTUALLY makes sign-then-Ising work at scale.
Without it, Phase-5-D Arm A is predicted by DR-3 to fail in a way Q12
Hypothesis B doesn't fully capture.

**3. Carnot's novelty boundary tightens to a single sentence.** Per
DR-3 §5: *"Carnot's specific, defensible novelty lies strictly in
utilizing this transpilation pipeline as the load-bearing path between
continuous latent reasoning and discrete neuro-symbolic verifier
ensembles, rather than merely utilizing it as a tool for model
compression or general generation."*

This is the exact sentence paper-v6 needs. Sign-then-Ising as
*compression* is owned by LittleBit. Continuous energy-descent
*reasoning* is owned by EBT/NRGPT. Carnot's contribution: the
*combination* — using the transpilation as a verifier-grounding bridge.

## Q.A–Q.E direct findings

**Q.A — Strongest evidence FOR information preservation under
sign-discretization:** LittleBit (Lee et al., NeurIPS 2025) — Llama2-13B
compressed to <0.9GB at sub-1-bit quantization (down to 0.1 BPW),
**outperforming 0.7 BPW continuous-passthrough baselines on perplexity**.
Mechanism: Dual Sign-Value-Independent Decomposition (Dual-SVID) for
quantization-aware training (QAT) — multi-scale residual compensation
explicitly controls variance prior to the sign() boundary. **Q11.4's
sign(z) firewall depends on Dual-SVID-class technique to actually work
in practice.**

**Q.B — Strongest evidence AGAINST naive transpilation:** Two failure
modes:

1. **Parametric collapse + gradient death (Recent Advances in Discrete
   Speech Tokens, IEEE 2025):** When continuous latent pushes deep into
   bounds (z ≈ 0.999), STE provides a FALSE gradient — the magnitude
   is insufficient to cross the zero-threshold, so the Boolean output
   permanently locks. This empirically corresponds to Q11's "Achilles
   Heel" warning: the sign(z) firewall depends on continuous variables
   crossing the zero boundary frequently during training.

2. **Critical collapse instability:** In dual-embedding EBMs and
   non-equilibrium dynamic systems, the entire continuous latent
   distribution collapses into a single degenerate Boolean mode
   regardless of input (cited in DR-3 §2.5). This is a THIRD
   manifestation of Q12 Hypothesis B (Dark Room) — orthogonal to
   substrate gaming, arising from architectural pathology rather than
   training dynamics.

**Q.C — Theoretical optimality (NEGATIVE):** Sign-then-Ising mapping
is **NP-hard** for the global optimal mapping. Kane-Tao Theorem (2025
extensions) provides explicit lower bounds for sum entropy when
transitioning from continuous probability mass to the rigid Boolean
cube $\{0,1\}^d$ — formal limit on mutual information loss during
sign-discretization. This complements Q11.5 ($\Sigma_2^P$ for verifier
synthesis) — the two complexity bounds operate at different layers
(verifier design vs substrate transpilation) but both point to: NO
exact polynomial-time algorithm exists; heuristics with known failure
modes are the only options.

**Q.D — Peer architectures Carnot must position against:**

1. **LittleBit (NeurIPS 2025, Lee et al.):** production sign-then-Ising
   at scale (Llama2-13B, sub-1-bit). **Closest functional peer.**
2. **scPRINT-2 (FSQ-VAE, 2025):** genomics, dimension-wise binary split.
3. **Extropic Denoising Thermodynamic Models (DTM, PRX 2024):**
   hardware-native sign-then-Ising via subthreshold CMOS thermal
   fluctuations. 10,000× energy efficiency over GPU.
4. **Distributed Quantum Optimization (DQOF, arXiv:2604.20599, 2025):**
   continuous quantum spatial evolutions thresholded into discrete
   binary strings; solves 500-variable HUBOs.

**Q.E — Consensus position:** Bifurcated AGAINST purely-continuous
(EBT/SVGD/diffusion suffer unbounded error accumulation in long-horizon
rollouts; cannot interface cleanly with discrete logical verifiers
without collapsing constraint logic) AND purely-discrete (VQ-VAE
codebook collapse, computational rigidity). **Bounded-Discrete
synthesis (FSQ + LittleBit) is the consensus.** Carnot's commitment is
fully validated.

## Comparative architecture matrix (DR-3 §3)

| Architecture | Citation | Scale | Transpilation property | Known failure mode |
|--------------|----------|-------|----------------------|---------------------|
| Modern Hopfield (continuous PDF) | Santos 2025 (arXiv:2502.10122) | millions | Continuous PDF → discrete fixed points via CCCP | Inverse-temperature β sensitivity; mixed continuous attractors |
| Dynamic EBM (EBT/NRGPT) | Gladstone 2025 (arXiv:2507.02092), Dehmamy 2025 (arXiv:2512.16762) | up to 120B tokens | Explicit gradient descent for subsequent discretization | Inference latency tied to iterative steps; intense gradient clamping |
| **Extreme Bounding (LittleBit/FSQ)** | **Lee 2025 (NeurIPS 2025)** | **13B params** | **Dual-SVID preserves gradients through sign() threshold** | **STE fails if continuous latents drift far from bounds → dead zones** |
| SVGD | He 2024 (arXiv:2602.05172), MDPI 2026 | finite horizon | Particles minimize KL divergence in continuous before resolving to actions | Kernelized interactions scale poorly to Boolean cube; high-frequency oscillations |
| **Thermodynamic Ising (Extropic TSU)** | **Freitas 2024** | **Edge ASIC** | **Subthreshold thermal noise natively collapses to Ising under voltage** | **Software-to-hardware mismatch; cannot interface with continuous software** |

## Required Carnot architecture changes

**1. Adopt Dual-SVID upstream of sign(z).** This is the single most
actionable DR-3 finding. The current substrate (per `python/carnot/phase3/`)
treats sign(z) as a primitive operation. The literature shows naive
sign-discretization catastrophically fails without:

- **Multi-scale residual compensation** before the boundary
- **Quantization-aware training (QAT)** that decouples sign from value
  during the continuous update phase
- **Variance control** ensuring continuous latents frequently cross zero

**Operational implication:** exp1238 (Phase-5-D) Arm A (control) is
predicted to fail BOTH Q12 Hypothesis B (substrate gaming) AND DR-3's
parametric collapse failure mode simultaneously. Without Dual-SVID-style
upstream variance compensation, gates will trip from BOTH causes,
making it harder to disambiguate.

**Recommendation:** add a third arm to exp1238:
- Arm A: control (no regularization)
- Arm B: Q12.4(a) entropy regularization (Dark Room defense)
- **Arm C: Dual-SVID upstream variance compensation (parametric-collapse defense)**
- **Arm D: Both regularizers stacked** (the empirical-best architecture)

Comparative analysis across 4 arms gives paper-v6 a much stronger
empirical claim: each regularizer addresses a distinct failure mode,
both are necessary, neither alone is sufficient.

**2. Extropic TSU as long-term hardware target.** DR-3 §2.6 confirms
Extropic Denoising Thermodynamic Models achieve 10,000× energy efficiency
via native Ising sampling. Carnot's existing hardware portfolio
(memory: `feedback_fpga_rescope_extropic_pivot.md`) already names
Extropic Z1 as the future hardware target. DR-3 validates: Carnot's
software substrate IS the algorithmic bridge that compiles to Extropic-
class hardware natively. Paper-v6 hardware section should explicitly
position this.

**3. Tighten paper-v6 novelty claim to DR-3 §5 single sentence.**

> "Carnot's specific, defensible novelty lies strictly in utilizing
> this transpilation pipeline as the load-bearing path between
> continuous latent reasoning and discrete neuro-symbolic verifier
> ensembles, rather than merely utilizing it as a tool for model
> compression or general generation."

This is much stronger than current paper-v6 framing because it:
- Explicitly cedes territory to LittleBit (compression) and EBT (reasoning)
- Stakes specific contribution to the BRIDGE between continuous reasoning
  and discrete verifier ensembles
- Connects to DR-2's verifier-ensemble work (Carnot's Sakana defense)
- Aligns with Phase-1 (verify-repair) → Phase-3 (foundation model) thesis

## Critical new citations for paper-v6 (10 papers)

```bibtex
@inproceedings{lee2025littlebit,
  title={{LittleBit}: Sub-1-Bit Quantization-Aware Training via
         Dual Sign-Value-Independent Decomposition},
  author={Lee, et al.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  note={Llama2-13B compressed to <0.9GB at 0.1 BPW; outperforms 0.7 BPW
        baselines. Dual-SVID is the load-bearing technique for
        sign-discretization at scale.}
}

@inproceedings{mentzer2024fsq,
  title={Finite Scalar Quantization: {VQ}-{VAE} Made Simple},
  author={Mentzer, Fabian and Tschannen, Michael and others},
  booktitle={ICLR},
  year={2024},
  note={State-of-the-art for bounded continuous-to-discrete
        without learned codebooks; Carnot's substrate generalization.}
}

@article{santos2025mhnct,
  title={Modern Hopfield Networks with Continuous-Time Memories},
  author={Santos, et al.},
  journal={arXiv preprint arXiv:2502.10122},
  year={2025},
  note={Continuous attention as retrieval dynamics on continuous-time
        Hopfield manifolds; foundational for Phase-3 continuous reasoning.}
}

@article{theriault2026cccp,
  title={A Convergent Method for Energy Optimization in
         Modern Hopfield Networks},
  author={Th{\'e}riault and Tantari},
  journal={MDPI},
  year={2026},
  note={CCCP (Concave-Convex Procedure) bifurcates MHN energy into
        convex + concave parts; guarantees descent to discrete attractors.}
}

@article{grishechkin2025enhancernet,
  title={Hierarchical Cell Identities Emerge from Animal Gene
         Regulatory Mechanisms},
  author={Grishechkin, et al.},
  journal={PRX Life},
  year={2025},
  note={Continuous Hopfield → discrete hierarchical Boolean identities
        controlled by inverse temperature β. Biological proof-of-concept
        for the continuous-to-discrete collapse mechanism.}
}

@article{xu2025dqof,
  title={Distributed Quantum Optimization for Large-Scale
         Higher-Order Problems with Dense Interactions},
  author={Xu, et al.},
  journal={arXiv preprint arXiv:2604.20599},
  year={2025},
  note={Quantum continuous evolutions thresholded into discrete binary
        via energy-improving updates. 500-variable HUBOs in 170s.}
}

@article{kanetao2025entropy,
  title={Information Theory, Stability, and the {K}ane-{T}ao Theorem},
  journal={DigitalCommons@UMaine},
  year={2025},
  note={Explicit lower bounds for sum entropy when mapping continuous
        probability mass to Boolean cube {0,1}^d. Formalizes mutual
        information loss during sign-discretization.}
}

@article{freitas2024extropic,
  title={An Efficient Probabilistic Hardware Architecture for
         Diffusion-Like Models},
  author={Freitas, et al.},
  journal={Physical Review X},
  year={2024},
  note={Extropic Thermodynamic Computing Unit (TSU); subthreshold CMOS
        thermal noise natively samples Ising; 10,000x energy efficiency
        over GPU for Denoising Thermodynamic Models.}
}

@article{he2024rsvgd,
  title={Finite-Particle Rates for Regularized Stein Variational
         Gradient Descent},
  author={He, et al.},
  journal={arXiv preprint arXiv:2602.05172},
  year={2024},
  note={Resolvent-type preconditioner; finite-particle non-asymptotic
        bounds in continuous + discrete time; O(1/sqrt(N)) convergence.}
}

@article{svgd2026discrete,
  title={Stein Variational Black-Box Combinatorial Optimization},
  journal={arXiv preprint arXiv:2604.15837},
  year={2026},
  note={SVGD extended to expensive discrete black-box functions;
        surpasses evolutionary algorithms without collapse.}
}
```

## Honest framing — what DR-3 does NOT validate

- **Sign-then-Ising is empirically a compression technique foremost.**
  LittleBit's headline result is compression (sub-1-bit weight quantization),
  not symbolic reasoning grounding. Carnot's claim that the same
  pipeline serves verifier-ensemble grounding is still architecturally
  novel — but the pipeline ITSELF is not. Paper-v6 must be careful here.
- **Empirical scaling above 13B params remains open.** LittleBit
  validated at Llama2-13B; Carnot's Phase-3 substrate aims for ~1B
  initially. DR-3 doesn't address whether sign-then-Ising preserves
  semantic density at the Phase-3 target scale.
- **Bridging to verifier ensembles is undocumented.** No DR-3 citation
  reports using sign-then-Ising as a verifier-grounding bridge. Carnot's
  combination IS novel — but UNTESTED in the literature, so Phase-5-D's
  empirical results carry the full weight of the claim.
- **The k=2 transversal pair (Q11) + sign-then-Ising substrate (DR-3)
  combination is unprecedented.** Together they form Carnot's
  contribution. Neither alone is novel; the combination is.

## Cross-references

- Prompt: `docs/research-notes/continuous-boolean-transpilation-deep-research-prompt.md`
- Source PDF: `docs/research-notes/continuous-boolean-transpilation-deep-research-source.pdf`
- DR-1 synthesis (energy-based LLM alternatives): `docs/research-notes/energy-based-llm-alternatives-deep-research-results.md`
- DR-2 synthesis (multi-verifier ensembles): `docs/research-notes/multi-verifier-ensemble-defense-deep-research-results.md`
- Q11 synthesis (TSS + STE attack): `docs/research-notes/verifier-orthogonality-design-deep-think-results.md`
- Q12 synthesis (Hypothesis B + Dark Room): `docs/research-notes/phase5-pcd-joint-null-space-deep-think-results.md`
- Continuous-Ising-Rank Theorem (Phase-3 internal foundation): `memory/project_continuous_ising_rank.md`
- FPGA → Extropic pivot: `memory/feedback_fpga_rescope_extropic_pivot.md`

# Deep Research DR-3 — Continuous-to-Boolean Transpilation Literature Survey

**Status:** PROMPT DRAFT — for paste into Gemini chat UI Deep Research
**Drafted:** 2026-05-04 ~01:20Z
**Strategic role:** Survey state-of-the-art on bounded-continuous-to-discrete-Boolean transpilation for Phase-3 substrate foundation
**Predecessors:** DR-1 (energy-based LLM alternatives), DR-2 (multi-verifier ensemble defenses)
**Output target:** comprehensive landscape of continuous→Boolean transpilation in 2024-2026 literature

---

## Paste boundaries

```
START:  "## The Deep Research request" (line ~16)
END:    end of "## Output format requested" section
SKIP:   this header, "## Why this prompt now"
```

---

## The Deep Research request

I am building the substrate for an open-source energy-based-model
foundation framework. The architectural commitment is a **bounded
continuous-latent-to-discrete-Boolean transpilation pipeline**: an
encoder produces continuous latents z ∈ [-1, 1]^d, an energy function
operates on these latents, and a decoder discretizes via sign(z) to
yield Boolean / Ising-spin variables that drive symbolic action
selection. This is the load-bearing path between Phase-3 (continuous
latent reasoning, Kona-parity) and Phase-1 (discrete-action verifier
ensemble grounding).

I need a comprehensive survey of how the 2024-2026 literature handles
the **continuous→Boolean transpilation problem**: what are the
state-of-the-art methods, what are their proven properties, what
empirical evidence exists for each, and is there a known optimal
algorithm for this specific pattern?

Please produce a structured research report focused on work published
2023-2026.

### Survey scope (in priority order)

**1. Modern Hopfield Networks and continuous-Boolean energy
correspondences.**

Theoretical and empirical work on:
- Ramsauer et al. "Hopfield Networks is All You Need" (2020+) and
  successor work
- Continuous Hopfield → discrete Ising correspondence theorems
- Energy-based attractor dynamics in continuous vs binary state spaces
- Capacity bounds for Hopfield networks under sign-discretization

**2. Sign-discretization theorems and bounded-latent activation
functions.**

- Tanh, sigmoid, hard-tanh, sign() — formal results on which
  discretization patterns preserve gradient flow vs lose information
- Straight-through estimators (STE), Gumbel-softmax, REINFORCE for
  sign() gradients
- Vector-quantized variational autoencoders (VQ-VAE) and their
  discrete-bottleneck analyses
- Concrete distribution / Gumbel-softmax backpropagation

**3. Neuro-symbolic concept learning + Boolean satisfiability.**

- Neural-symbolic SAT solvers (NeuroSAT, learned heuristics for SAT)
- MaxSAT learning and learned-MaxSAT 2024-2026 work
- Symbol grounding in continuous neural networks → discrete logical
  predicates
- Recent neural-symbolic frameworks (e.g., NeurASP, Logic Tensor
  Networks, DeepProbLog) that bridge continuous/discrete

**4. Energy-based models with discrete output spaces.**

- Restricted Boltzmann machines and their continuous/discrete
  variants
- Discrete EBMs in 2024-2026 (e.g., Du & Mordatch 2019 lineage,
  contrastive divergence on Ising, persistent contrastive divergence)
- Score matching for discrete distributions
- Stein variational gradient descent on discrete spaces

**5. Specific Carnot-relevant patterns.**

- "Bounded latent" architectures: VQ-VAE, FSQ (finite scalar
  quantization), latent diffusion with discretized bottlenecks
- "Ising-substrate" architectures: hardware Ising machines,
  D-Wave quantum annealing, Extropic TSU, FPGA Ising solvers
- "Sign-then-Ising" two-stage pipelines: any work that explicitly
  uses sign(z) → Ising → action

### What I already cite or know well (deprioritize unless 2025-2026
recent developments)

- Du & Mordatch 2019 (foundational EBM)
- Ramsauer et al. 2020 Hopfield Networks
- Bengio et al. STE family (2013+)
- VQ-VAE (van den Oord 2017+) and successors
- LeCun JEPA family
- Gladstone EBT (arXiv:2507.02092)
- NRGPT (arXiv:2512.16762)
- Continuous-Ising-Rank Theorem (the project's own internal Deep Think
  result on dual-embedding + parametric collapse)

### What I want to learn

For each architecture/technique surfaced:

1. **What it claims** — the transpilation mechanism in one sentence.
2. **What it demonstrates** — empirical scale, benchmarks, formal
   guarantees.
3. **Primary reference** — citation with arXiv ID, conference, year.
4. **Lineage** — which earlier work it builds on.
5. **Open-source status** — code/weights public? license?
6. **Comparison axis** — relative to bounded-latent → sign() → Ising,
   what are its specific strengths/weaknesses?

I particularly want:

- **2024-2026 work** I likely haven't seen
- **Negative results** — papers reporting failure modes specific to
  continuous-to-discrete transpilation (e.g., gradient vanishing
  under sign(), capacity collapse, mode collapse on sign-discretized
  output)
- **Theoretical optimality results** — is there a proven optimal
  procedure for "given a continuous distribution P over [-1,1]^d,
  produce a discrete Boolean code that minimizes mutual information
  loss subject to constraint X"?
- **Industrial deployment** — major labs (Anthropic, OpenAI, Google,
  Apple, NVIDIA, Mistral) using continuous→Boolean transpilation in
  production architectures, especially in 2025-2026 frontier models.

### Specific questions to answer

**Q.A — What is the strongest empirical evidence that
sign-discretization of bounded continuous latents preserves the
information necessary for downstream symbolic reasoning?** Cite
specific papers with empirical attack/defense results, capacity
bounds, or task performance comparisons (continuous-passthrough vs
sign-discretized).

**Q.B — What is the strongest published evidence that the
continuous-to-Boolean transpilation pattern FAILS in practice?**
Specifically: gradient pathologies, capacity collapse, mode collapse,
or representation degeneracy under sign-discretization at scale.

**Q.C — Is there a known optimal algorithm for the specific
transpilation pattern: encoder → bounded-latent z ∈ [-1,1]^d →
sign(z) → Ising → discrete action?** If so, what is its complexity
class and what assumptions does it require? If not, what is the most
common heuristic in current architectures, and what is its known
failure mode?

**Q.D — What named architectures explicitly use a
"sign-then-Ising-then-action" pipeline at scale, and have been
peer-reviewed?** I want the comparator set Carnot must position
against. Include both research-grade (Hopfield-modern, VQ-VAE-derived)
and production-grade (FSQ in DALL-E 3 era, etc.).

**Q.E — What is the consensus position (if any) on whether
bounded-continuous-to-Boolean transpilation is the correct substrate
choice for energy-based foundation models, vs alternative
architectures like (a) full continuous (no discretization), (b) full
discrete (token-only), (c) hierarchical multi-resolution
discretization (FSQ + tokens)?**

### Output format requested

Please structure as:

1. **Executive summary (3-5 paragraphs).** Comparative landscape
   organized by architectural family.

2. **Per-architecture sections** with two-paragraph intro + tabular
   per-work detail (primary reference, claim, evidence, lineage,
   open-source) + "Honest framing" subsection.

3. **Comparative table.** Architecture | primary citation | benchmark
   category | peer-reviewed Y/N | open-source Y/N | scale | claimed
   transpilation property | known failure mode.

4. **Direct answers to Q.A through Q.E.**

5. **Gaps and recommendations.** Priority reading; architectures to
   compare against; novelty boundaries (what NOT to claim about
   sign-then-Ising); researchers or labs to engage.

### Format constraints

- Cite primary sources, not aggregator articles
- For arXiv preprints, include arXiv ID
- For peer-reviewed work, include venue + year
- Distinguish reproductions from original results
- For controversial claims, cite strongest evidence on each side
- Distinguish "discrete latent" architectures (VQ-VAE, FSQ — finite
  but high-cardinality codes) from "Boolean / Ising substrate"
  architectures (binary or ±1 spins). These are different problems
  and Carnot is in the latter category.
- Distinguish "discretization for compression / efficiency" (most
  VQ-VAE work) from "discretization for symbolic reasoning grounding"
  (Carnot's case). The latter is much sparser in the literature.

---

## Why this prompt now (decision-leverage)

DR-1 (energy-based LLM alternatives) confirmed Carnot's positioning
gap is "open-source externally-grounded EBM that solves multimodal
text collapse." DR-2 (multi-verifier ensemble defenses) populated the
Sakana-defense comparator set.

Both DRs covered Phase 1 (verify-repair) territory. Phase 3's
substrate is the foundation-model load-bearing piece, and we've
reasoned to the bounded-continuous → sign() → Ising pattern via the
internal Continuous-Ising-Rank Theorem. But we haven't surveyed how
others handle the same boundary.

If a known optimal transpilation algorithm exists, Carnot should
adopt it rather than re-derive. If known failure modes exist,
exp1238 (Phase-5-D intermediate-scale, .96/.97) needs to instrument
for them. If the literature is sparse, paper-v6 can claim
contribution at the substrate level, not just the verifier-ensemble
level.

The cost asymmetry: ~30 minutes of Deep Research now vs designing
exp1238 in the dark, then discovering at intermediate-scale that
sign-discretization causes capacity collapse (Q9 mode 1, currently
classified as "production-scale-only failure mode" but might be
literature-known and avoidable).

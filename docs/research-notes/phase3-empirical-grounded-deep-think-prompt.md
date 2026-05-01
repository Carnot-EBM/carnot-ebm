# Deep Think Prompt — Phase-3 Architecture Path Given Mixed Empirical Evidence

**Status:** Ready to send. Asks Deep Think to integrate today's
empirical findings (2026-05-01) into a rigorous decision tree for
the Phase-3 prototype, replacing the purely-theoretical analysis
that drove the 2026-04-30 architecture decision.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The 2026-04-30 blind-spot audit identified 5 FATAL findings against
the originally-proposed DBAE-EBM (Deterministic Bounded Autoencoder +
Latent EBM) architecture for Phase-3 of the Carnot project:

- **Finding #1 (dimensionality guillotine):** the boolean latent
  space `z ∈ {-1,+1}^d` cannot represent the continuum of valid
  outputs the encoder is asked to reconstruct.
- **Finding #2 (synchronous Glauber non-equilibrium):** parallel
  Glauber dynamics on arbitrary symmetric J does not preserve
  detailed balance, so the FPGA hardware sampler does not sample
  from `exp(-E(z)/T)`.
- **Finding #3 (Hopfield capacity collapse):** the latent EBM's
  attractor capacity scales as ~0.14·d and saturates well below
  the dimensionality the encoder produces.
- **Finding #4 (Taylor spurious corners):** the bounded
  autoencoder's smooth decoder admits constraint-pass corners that
  the Ising energy cannot penalize.
- **Finding #5 (higher-order logic eradication):** the
  `sgn(z)→Ising` projection collapses any clause structure with
  arity > 2 into pairwise terms, eradicating the higher-order
  logic the verifier suite is supposed to enforce.

**As of 2026-05-01, two of these have been addressed by direct
empirical work, and one cross-cutting precondition has been
empirically refuted:**

1. **Finding #2 is EMPIRICALLY CONFIRMED.** Experiment 1094 (Phase
   2a Sampler Correctness Audit) measured `KL(P_parallel_glauber
   || P_correct_gibbs) = 3.07` on a 12-spin frustrated
   antiferromagnetic ring, against the 0.05 acceptance threshold.
   The verdict was `fpga_sampler_distribution_mismatch_confirmed`.
   The KV260 FPGA hardware path therefore samples from a
   distribution ~3 nats away from the model's intended Boltzmann
   distribution. Detailed-balance verified on 2-spin closed form;
   instrumentation is `KLDivergenceEstimator` from
   `python/carnot/eval/diagnostics.py` (exp1090).

2. **Phase-1c precondition for AND-composition is EMPIRICALLY
   FAILING.** Experiment 1093 (Phase 1c Verifier Joint Null-Space
   Measurement) produced two key numbers: joint null-space
   fraction = 0.0 (good — no input fools all 3 verifiers at once)
   AND maximum pairwise r-correlation = 0.66 across the 3 Tier-0
   text probes (NUP / SpilledEnergy / KAN), against the 0.5
   diversity threshold. The verdict was
   `verifiers_correlated_diversity_needed`. The Phase-3 architecture
   assumes AND-composition over k=15+ verifiers exponentially
   shrinks the joint null space; this assumption requires verifier
   diversity that the current 3-probe suite has not achieved.

3. **Findings #1, #3, #4, #5 remain THEORETICALLY UNVALIDATED.**
   No empirical instrumentation has been built for them.

**One additional empirical finding may be relevant:** experiment
1100 (Cascade Validation on SOTA Model Outputs) reported verdict
`cascade_validated_sota_inefficient` — the cascade does discriminate
correct vs incorrect SOTA outputs but with the cost structure
making the AND-composition Pareto-suboptimal for the marginal cost
of additional verifier evaluations.

### Established design constraints

The Phase-3 prototype must obey:

- **CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial
  Check Discipline" (2026-04-30 mandatory):** every phase prototype
  must include a software prototype + measurable pass/fail tests +
  hostile-reviewer round before scaling. Theoretical rigor alone
  has been shown insufficient (the audit caught 5 FATAL findings
  three rigorous Deep Think rounds missed).

- **CLAUDE.md "Decentralization-Respecting Design Constraints":**
  local-first using open models always; closed-frontier-model
  integration optional, never required; multiple integration
  surfaces in parallel (Python API, CLI, MCP, HTTP REST); hardware
  portability across GPU (CUDA primary) + NPU (sovereignty anchor)
  + WebGPU + emerging Extropic Z1; no vendor-specific abstractions
  in the core verifier stack.

- **Phase-2 hardware status (2026-04-30 user directive):** FPGA
  has been re-scoped to proof-of-concept tier. Production hardware
  is now Extropic Z1 + photonic. KV260 is preserved as engineering
  proof-point but is NOT load-bearing for production claims.

### The decision question (please answer rigorously)

Given the empirical state above, design a rigorous Phase-3 prototype
specification that integrates BOTH (a) the empirical refutation of
the parallel-Glauber sampler AND (b) the empirical r=0.66
verifier-correlation finding. Specifically address:

#### Q1. AND-composition feasibility ceiling for k

Given pairwise r-correlation r=0.66 across k=3 text probes, what is
the realistic upper bound on k achievable with topologically
distinct verifier mechanisms? Specifically:

- Which verifier mechanism families produce r-correlation < 0.5
  with respect to the 3 existing text probes (NUP, SpilledEnergy,
  KAN-based SOSKANEnergyV3)? Candidates to evaluate: Z3-AST formal
  verification, semantic-embedding probe, ThinkPRM step-level
  probe, output-format probe, runtime-execution probe (testing
  generated code), combinatorial-encoding probe (via Ising/Potts
  ground state).
- Is there a theoretical argument (e.g., information-theoretic
  bound on independent feature extraction from a fixed input
  distribution) that places k at ~5-7 versus the architecturally-
  assumed k=15? If so, derive the bound.
- If the achievable k is bounded below 15, what is the
  Friedrichs-angle / joint-null-space-fraction tradeoff? Does
  AND-composition over k=5 with r-correlation < 0.3 give better
  guarantees than k=15 with r-correlation = 0.55?

#### Q2. FPGA Glauber sampler architectural rescue

Given Finding #2 is empirically confirmed (KL=3.07), rank-order the
following sampler architectures on (a) detailed-balance correctness,
(b) hardware speedup vs CPU, (c) problem-class restriction, (d)
fit with the existing exp1098 Potts-machine bitstream and the
exp1041/exp1068 Ising bitstream:

1. **Sequential single-spin Glauber** — correct on arbitrary J,
   loses parallelism, what is the realistic latency on KV260?
2. **Bipartite-block parallel Glauber** — correct on bipartite J
   only, restricts the encoder's expressible J matrices to a
   strict subset. Is the bipartite restriction empirically
   compatible with the 6 existing constraint families
   (constraint store, code AST, numeric SAT, etc.)?
3. **Metropolis-Hastings** — correct on arbitrary J via accept/
   reject, requires a hardware random gate sampler and a hardware
   ratio comparator. What is the FPGA gate-count overhead?
4. **Heat-bath continuous-time chain** — correct in continuous
   time, requires asynchronous hardware. Compatible with KV260's
   AXI-Lite synchronous register interface?
5. **Hamiltonian Monte Carlo on relaxed boolean** — train the
   latent space to be smooth (continuous), use HMC. Re-introduces
   the dimensionality concern from Finding #1; does that
   self-cancel or compound?
6. **Abandon FPGA Glauber, redirect Phase-2 budget to Extropic Z1
   access** — what is the production-hardware path's risk profile
   (Z1 silicon availability, SDK maturity, tooling)? What are the
   one or two highest-impact Z1-vendor-relationship tasks the
   Phase-2 milestone should propose?

For each of (1)-(6), provide: (a) empirical pass/fail criterion
the prototype must meet, (b) instrumentation plan (which classes
from exp1090's diagnostics.py to use, what new ones to add), (c)
expected wall-clock cost in milestones at the project's current
~2-week cadence.

#### Q3. Phase-3 prototype training path

Given Q1 (AND-composition feasibility ceiling) and Q2 (sampler
architecture choice), specify the Phase-3 prototype training
recipe. Address explicitly:

- **Training corpus:** what corpus characteristics avoid the
  exp1099 RLVR+SSD failure mode (pre-filtered corpus has all-zero
  energy → energy filter is degenerate)? Does the training corpus
  need to include both pre- and post-verifier-filtered samples to
  preserve the energy gradient signal?
- **Verifier suite at training time:** if Q1 concludes the realistic
  k is 5-7, must Phase-3 deploy with the achievable k AND the
  current correlation r, or must Phase-1d (verifier diversity
  expansion) precede Phase-3 prototype work?
- **Sampler choice at training time:** if Q2 concludes the corrected
  FPGA sampler is one of 1-5, does Phase-3 prototype run on FPGA or
  CPU during training? At what point does it migrate?
- **The 3-stage training schedule** from the original Round-3 design
  (DBAE pre-train → EBM training → joint fine-tune): does it
  survive the empirical findings? Or does the EBM training stage
  need to be replaced with something accommodating the smaller-k
  composition?

#### Q4. Path to Phase-3 published claim

What is the minimum empirical state at which Phase-3 produces a
claim worth a position-paper-v3 update (i.e., ARM-EBT-parity claim
with empirical evidence)? Specifically:

- Is `final_energy = 0` on a held-out test corpus + measurable KL
  divergence vs the model's intended Boltzmann the right pair of
  acceptance metrics?
- What does `parity` operationalize as? Token-level decoded-text
  diversity matching ARM perplexity? AUROC on a held-out OOD
  benchmark? Sample-quality on combinatorial generative tasks
  (program synthesis, theorem proving)?
- What is the smallest model size at which the Phase-3 prototype
  can demonstrate non-degenerate behavior? Can it be 0.8B
  parameters (matching Carnot's existing Qwen3.5-0.8B smoke-test
  tier), or does it need the SOTA tier (Qwen3.6-35B-A3B-GGUF or
  Gemma-4-31B-it-GGUF)?

### What I am NOT asking

- I am NOT asking for new theoretical defense layers. The
  derivation chain through Phase-7 is complete (Stochastic-Veto
  Continuum Memory). The question is how to ship the static Phase-3
  empirically, not how to extend the theoretical defense.
- I am NOT asking for Phase-1c verifier-diversity-expansion
  recommendations specifically (that's Q1's input, not its output).
- I am NOT asking to revisit the FPGA re-scope decision. Phase-2
  hardware is now Extropic Z1 + photonic per user directive; FPGA
  is POC-only. Q2 must respect this.

### Output format

Please structure the response as:

1. **Executive summary** (1-2 paragraphs naming the recommended
   Phase-3 prototype specification)
2. **Q1 answer** (AND-composition feasibility ceiling, with
   theoretical bound + empirical recommendations)
3. **Q2 answer** (ranked sampler architectures with criteria-by-
   criteria evaluation)
4. **Q3 answer** (training recipe with corpus / verifiers / sampler /
   stages)
5. **Q4 answer** (minimum empirical state for position-paper claim)
6. **Risk register** — the top 5 ways this Phase-3 prototype could
   silently fail empirically (analogous to the 5 FATAL findings the
   2026-04-30 audit caught), each with the diagnostic instrumentation
   that would surface it.

### Honesty requirement

Per CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial
Check Discipline": if the empirical evidence does not support a
deployable Phase-3 prototype yet, say so explicitly. Recommend the
specific empirical work that must precede Phase-3 prototype
construction. The honest "not yet ready" answer is more valuable
than a speculative "yes ship it" answer.

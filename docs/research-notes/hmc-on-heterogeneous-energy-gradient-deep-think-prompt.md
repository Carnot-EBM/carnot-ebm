# Deep Think Prompt — HMC Compatibility on Carnot's Heterogeneous-Lipschitz Energy Gradient

**Status:** Ready to send. Highest-priority Deep Think question for the
sharpened Phase 4 plan. The Phase-4 prototype task
(`exp11XX-hmc-sampler-on-carnot-ebm`) assumes Carnot's existing
`∇E` is HMC-compatible. If the gradient's heterogeneous Lipschitz
behavior makes HMC pathological, the whole prototype falls before
it starts.
**Date drafted:** 2026-05-02 07:25Z
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project (open-source Apache 2.0, energy-based verification
of LLM outputs) is preparing a Phase-4 inference-mode extension that
swaps the current Langevin/Gibbs sampler for Hamiltonian Monte Carlo
(HMC). Motivation: a recent deep-research document established the
mathematical equivalence

```
score(x) = ∇_x log p(x) = -∇_x E(x)   [for Boltzmann-distributed EBM]
```

so the same energy gradient `∇E` that drives Carnot's existing
verify-repair loop also drives HMC sampling and diffusion-of-thought
reverse denoising. The Themesis Seed IQ system has demonstrated 100%
on ARC-AGI-3 with 115% human action-efficiency using Hamiltonian
Monte Carlo on a "Field of Belief" energy landscape (verified via
public video demonstration); Phase 4 tests whether HMC on Carnot's
existing energy landscape reproduces that result.

### Carnot's energy landscape

Carnot's production energy function (post-exp1128, AUROC=0.94 on
FoVer eval) is a sum of contributions from a heterogeneous k=5
AND-composed verifier ensemble:

```
E(x, y) = w_Z3 · E_Z3(x, y)
        + w_AST · E_AST(x, y)
        + w_Sem · E_Sem(x, y)
        + w_PRM · E_ThinkPRM(x, y)
        + w_JSON · E_JSON(x, y)
```

The five components have **substantially different mathematical
character**:

1. **Z3-AST formal verifier (`E_Z3`):** symbolic constraint solver.
   Gradient `∇E_Z3` is a finite-difference approximation over
   discrete propositional changes. Potentially discontinuous at
   constraint-violation boundaries.

2. **AST structural verifier (`E_AST`):** parses the candidate as
   AST, scores tree structure compliance. Gradient is finite-
   difference over token-level edits. Discrete-discontinuous.

3. **Semantic embedding probe (`E_Sem`):** continuous neural-network
   probe (transformer hidden states + linear head). Gradient is
   smooth, computed via standard autograd. Lipschitz-bounded by
   the transformer's layer norms + spectral-norm regularization.

4. **ThinkPRM step-level probe (`E_ThinkPRM`):** generative process
   reward model. Gradient is autograd over the verbalized
   verification chain-of-thought. Mid-Lipschitz: smooth where
   the model is decisive, can spike near boundary cases.

5. **JSON schema validator (`E_JSON`):** symbolic schema-conformance
   check. Gradient is finite-difference over candidate-text edits.
   Discrete-discontinuous like Z3 and AST.

### The HMC compatibility concern

Hamiltonian Monte Carlo's efficiency advantage over Langevin
dynamics comes from momentum-augmented dynamics that traverse the
energy landscape via leapfrog integration of the Hamiltonian
`H(x, p) = E(x) + p^T M^{-1} p / 2`. HMC's correctness and
efficiency depend on:

- **Lipschitz-bounded gradient** (or at least: gradient bounded
  almost everywhere, with measure-zero exceptions for jumps)
- **Smooth-enough log-density** that leapfrog integration is
  numerically stable at reasonable step sizes
- **Roughly-uniform Lipschitz constant** across the gradient's
  components (or per-component preconditioning)

Carnot's `∇E` may violate all three. The three discontinuous
symbolic components (Z3, AST, JSON) contribute jumps; the two
smooth components (Sem, PRM) contribute different-scale smooth
flows. A naive HMC implementation could:

- **Reject most proposals** (acceptance rate near zero) due to
  large `H` errors at constraint-violation jumps
- **Get stuck in basins** that the smooth components define but
  the discrete components fence off
- **Produce biased samples** if leapfrog skips over discontinuities

Seed IQ's claimed ARC-AGI-3 success suggests SOME formulation of
HMC-on-energy-landscape works; we need to know whether Carnot's
specific `∇E` formulation is in the working regime.

### Specific questions

1. **What's the methodology for measuring whether a sum of
   heterogeneous-Lipschitz gradients is HMC-compatible?** Specify
   diagnostics that can be computed on Carnot's existing post-
   exp1128 energy function (AUROC=0.94) without launching the full
   prototype. Each diagnostic should:
   - Test a specific aspect of HMC compatibility (acceptance rate,
     trajectory closure, autocorrelation, ESS)
   - Use only existing infrastructure (k=5 verifier ensemble plus
     a small synthetic test corpus, ~100 examples)
   - Have a directional threshold (no specific numerical values
     per the Carnot prediction-error pattern)

2. **What three regimes should the diagnostics distinguish?**
   - **A: HMC works directly** — `∇E` is well-conditioned enough
     that off-the-shelf NumPyro HMC (with default leapfrog and
     adaptive step-size) gives high acceptance and low
     autocorrelation. No preconditioning needed.
   - **B: HMC works with per-component preconditioning** — the
     five gradient components have different scales but are
     individually well-behaved. Adding a per-component step-size
     mass matrix `M` recovers HMC efficiency.
   - **C: HMC is fundamentally inappropriate** — the symbolic
     components' discontinuities make Hamiltonian dynamics ill-
     defined; the trajectory `H` errors are unbounded; acceptance
     rate stays near zero regardless of step size or
     preconditioning. Fall back to Langevin with adaptive step
     size, or to surrogate-gradient HMC, or to a blocked Gibbs
     sampler that handles the symbolic components separately.

3. **For each regime**, what would the prototype's first 100 HMC
   steps look like? Specify the empirical signature an operator
   would observe in the first ~10 minutes of running the
   prototype, so we can detect the regime quickly without running
   the full ARC-AGI-3 evaluation.

4. **If regime B (preconditioning needed) is the answer**, what's
   the mathematical principle for choosing the per-component mass
   matrix? Don't give a specific recipe — describe the criterion
   that distinguishes "good preconditioner" from "preconditioner
   that masks the problem rather than solving it".

5. **If regime C (HMC inappropriate) is the answer**, what's the
   most-promising fallback? Three named candidates:
   - **Langevin with adaptive step size** (Nadam-on-energy or
     similar) — same gradient, no momentum, smaller step at
     discontinuities
   - **Surrogate-gradient HMC** (replace symbolic-component
     gradients with a smooth surrogate trained to match the
     symbolic component's verdict on labeled data)
   - **Blocked Gibbs / Metropolis-within-Gibbs** (continuous
     components via HMC, symbolic components via separate
     Metropolis steps)
   For each, what diagnostic on Carnot's existing data would tell
   us whether it's the right fallback?

### What NOT to recommend

- **No specific numerical values** for step sizes, mass matrix
  entries, leapfrog steps, acceptance-rate thresholds, autocorrelation
  bounds. The Carnot prediction-error pattern (memory:
  `feedback_carnot_prediction_pattern.md`) makes these systematically
  wrong. Stay in the diagnostic-design / regime-classification lane.
- **No paper-citation-only answers** (e.g., "see Neal 2011"). The
  question is about Carnot's specific `∇E` configuration, not
  general HMC theory.
- **No recommendations that require modifying the verifier
  ensemble**. The k=5 ensemble is production-wired (post-exp1121,
  exp1128 calibration). The HMC sampler must work on this `∇E` or
  fall back to a non-HMC method.

### Output format request

```
DIAGNOSTICS:
  Diagnostic 1: <name>
    Tests: <which HMC compatibility property>
    Data: <which existing artifacts + synthetic corpus>
    Quantity: <formula or pseudocode>
    Direction: <high/low value indicates compatibility>
    Confidence calibration: <when this diagnostic is less reliable>
  Diagnostic 2: ...
  ... (3-5 diagnostics)

REGIME CLASSIFICATION:
  Diagnostic combination → regime A
    Empirical signature in first 100 steps: <what operator sees>
  Diagnostic combination → regime B
    Empirical signature: ...
    Preconditioning principle: <criterion for good M>
  Diagnostic combination → regime C
    Empirical signature: ...
    Recommended fallback: <Langevin / surrogate-gradient / blocked Gibbs>
    Diagnostic for fallback: <which test confirms the right fallback>

UNRESOLVABLE UNCERTAINTY:
  <if any aspect of Carnot's ∇E configuration cannot be diagnosed
   without running the full prototype, name it explicitly>
```

### Cross-validation reminder

Per `feedback_carnot_prediction_pattern.md`: prior Deep Think rounds
have qualitative survival claims well-calibrated, but specific
numerical prescriptions systematically wrong. This question is in
the diagnostic-design / regime-classification lane. If your answer
drifts toward parameter prescriptions (step size = X, mass matrix
diagonal = [a, b, c, d, e]), please flag the drift explicitly and
provide the qualitative answer alongside.

Note: Q4 (verifier degradation gate methodology) self-flagged its
only soft drift on `min_transient_confidence`; Q5 (Phase-3 substrate
contamination) self-flagged TWO parameter prescriptions
(rotation rate, k_held_out). The same self-discipline is welcome
here.

Note also: this is the 6th Deep Think round in 24 hours. The prior
five all stayed in the methodology / compositional-analysis lane
without parameter drift. The Q7 question is more empirically-grounded
than the prior five (Carnot has actual `∇E` from a working production
ensemble), so the diagnostics should be specifically computable on
Carnot's data, not generic HMC-theory results.

---

## End of prompt

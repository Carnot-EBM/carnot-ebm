# Deep Think Prompt — Discrete-Action Representation for HMC Sampling on ARC-AGI-3

**Status:** Ready to send. Second-priority Deep Think question for the
sharpened Phase 4 plan, after Q7 (HMC compatibility on heterogeneous
∇E). The Phase-4 prototype's empirical comparison to Seed IQ depends
on the **action representation** Carnot uses to bridge HMC's
continuous sampling with ARC-AGI-3's discrete action space.
**Date drafted:** 2026-05-02 07:35Z
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's Phase-4 prototype (`exp11XX-hmc-sampler-on-carnot-ebm`)
extends an existing energy-based verification pipeline by swapping
the current Langevin/Gibbs sampler for Hamiltonian Monte Carlo
(HMC). Motivation: the Themesis Seed IQ system has demonstrated
100% on ARC-AGI-3 with 115% human action-efficiency using HMC on
a continuous "Field of Belief" energy landscape. The prototype tests
whether HMC on Carnot's existing energy landscape (k=5 verifier
ensemble + DBAE-EBM, AUROC=0.94 post-exp1128) reproduces that
result.

### The discrete-continuous mismatch

ARC-AGI-3 is an interactive, turn-based benchmark where each task
is a video-game-like environment. Agents take **discrete actions**:

- Place a tile at grid position (i, j)
- Move an object in a cardinal direction (up / down / left / right)
- Apply a labeled transformation rule
- Confirm a hypothesis-resolution

The action space at each turn is finite and structured: it's a
union of discrete categories, each with discrete-grid coordinates,
and each action is either valid or invalid given the current
environment state. There is no continuous action representation
in the benchmark itself.

Hamiltonian Monte Carlo, however, is **fundamentally a continuous
sampler**. It evolves a state `x ∈ R^n` along Hamiltonian
trajectories `H(x, p) = E(x) + p^T M^{-1} p / 2` via leapfrog
integration. The momentum-augmented dynamics require `x` and `p`
to live in a continuous space with a smooth Riemannian metric.

Carnot must therefore choose how to represent ARC-AGI-3 actions
in HMC's continuous state space. The two textbook options are:

#### Option A: Continuous relaxation

Embed the discrete action set `A` in `R^n` (e.g., n = |A|, with
each action a basis vector or a learned embedding). Run HMC on
the continuous embedding. At decision time, snap the sampled
continuous state to the nearest valid discrete action via
nearest-neighbor or argmax.

- **Pros:** straightforward HMC on standard manifolds; gradient
  is well-defined everywhere in the embedding space.
- **Cons:** can produce continuous states that snap to *invalid*
  actions (constraint violations); the snap step breaks the
  Hamiltonian's energy conservation, requiring careful
  Metropolis correction.

#### Option B: Bayesian categorical (simplex HMC)

Maintain a continuous posterior over action *probabilities*
`p(a | x) ∈ Δ^{|A|-1}` (the probability simplex). Sample via
HMC on the simplex (e.g., via the softmax parameterization or
the Stiefel-manifold embedding).

- **Pros:** valid action probabilities everywhere by construction;
  HMC samples are interpretable as Bayesian beliefs.
- **Cons:** simplex HMC can get stuck on the boundary (probability
  collapses to a vertex); the simplex has no isotropic Riemannian
  metric, so leapfrog requires geodesic integration; the posterior
  can become multimodal in ways HMC handles poorly.

### Seed IQ's claim and the third-option question

The Themesis Seed IQ system claims to handle ARC-AGI-3's discrete
action space *natively* via what they call "topological field
cognition" with HMC on the "Alpha-Omega Field of Belief" (AΩ FoB
HMC). Their reported numbers (per public demonstration video and
recent deep-research documents):

| Sub-game | Human actions | Seed IQ actions | Efficiency |
|---|---|---|---|
| **VC33** | 307 | 173 | 43% fewer |
| **FT09** | 163 | 75 | 54% fewer |
| **LS20** (novel complexity) | 546 | 433 | 21% fewer |

Seed IQ doesn't explicitly describe its action representation as
either Option A (relaxation + snap) or Option B (simplex). The
public material describes the cognition as a "continuous geometric
field" where:

- Multiple structural hypotheses coexist as a "quantum superposition"
- Free-energy minimization causes a "binary phase transition" that
  collapses the field to a discrete answer
- The agent infers environmental priors (invariances, constraints)
  rather than searching action sequences

This sounds like **a third option**: HMC on a representation where
discrete actions emerge from the dynamics of the continuous field
rather than being snap-to-discrete after the fact. The "phase
transition" language suggests a **bifurcation** in the energy
landscape itself, not a post-hoc rounding step.

### What we want from Deep Think

We are NOT asking which option Seed IQ uses (their architecture is
proprietary). We ARE asking:

1. **Is the third option (action emerges from field dynamics)
   mathematically distinct from Options A and B**, or does it
   reduce to one of them under analysis? If distinct, what's the
   formal name and core construction?

2. **For each of the three options**, what's the failure mode
   that would make it inappropriate for Carnot's specific energy
   landscape (heterogeneous Lipschitz, k=5 ensemble, see Q7)?

3. **Which option is most compatible with Carnot's existing
   infrastructure**? Specifically:
   - The k=5 verifier ensemble is already wired to score (text,
     candidate-output) pairs, not (state, candidate-action) pairs
   - The DBAE-EBM (Phase 3) operates on a bounded latent
     `z ∈ [-1, 1]^d`
   - The action space for ARC-AGI-3 is at most |A| = O(grid_size² ×
     n_action_types), typically ≤ 1000 discrete actions per turn

4. **What diagnostics on Carnot's existing data** would tell us
   whether the chosen option preserves action validity at >95% of
   sampled states? (Without running the full ARC-AGI-3 evaluation.)

5. **What's the empirical signature** that distinguishes "the
   action representation is working" from "the action representation
   is producing high-energy actions that the verifier ensemble
   then rejects" (i.e., Option A's snap-to-invalid failure mode)?

### Constraints on output

- **NO specific embedding dimensions, simplex parameterizations,
  or bifurcation thresholds.** The Carnot prediction-error pattern
  (memory: `feedback_carnot_prediction_pattern.md`) makes specific
  numerical prescriptions systematically wrong. Stay in the
  representational-choice / failure-mode-analysis lane.
- **NO recommendations that require Themesis-proprietary
  information.** If you reason about Seed IQ's likely architecture,
  derive it from public information (Karl Friston's Free Energy
  Principle, Renormalising Generative Models, AΩ FoB HMC public
  description) only.
- **DO acknowledge the three options as a complete classification.**
  If there's a fourth option we haven't named, identify it
  explicitly. If the three options are exhaustive (modulo
  isomorphisms), say so.

### Output format request

```
THIRD OPTION ANALYSIS:
  Is "action emerges from field dynamics" mathematically distinct
  from Options A and B? <yes / no / partial>
  If distinct: <formal name + core construction>
  If reduces to A or B: <which one + under what limit>

THE THREE (OR FOUR) OPTIONS:
  Option A (continuous relaxation):
    Failure mode for Carnot's heterogeneous ∇E: <what breaks>
    Failure mode signature: <what operator would observe>
    Diagnostic: <which test confirms the failure mode>
    Compatibility with Carnot infrastructure: <high / medium / low + why>
  Option B (simplex HMC):
    Failure mode for Carnot's heterogeneous ∇E: <what breaks>
    Failure mode signature: <what operator would observe>
    Diagnostic: <which test confirms the failure mode>
    Compatibility with Carnot infrastructure: <high / medium / low + why>
  Option C (field dynamics):
    [if distinct from A and B]
    Mathematical construction: <core idea>
    Failure mode for Carnot's heterogeneous ∇E: <what breaks>
    Failure mode signature: <what operator would observe>
    Diagnostic: <which test confirms the failure mode>
    Compatibility with Carnot infrastructure: <high / medium / low + why>

RECOMMENDATION FOR CARNOT'S PHASE-4 PROTOTYPE:
  Most-compatible option: <A / B / C>
  Reason: <which Carnot-specific property (heterogeneous Lipschitz,
           bounded latent, k=5 ensemble structure, ≤1000 actions
           per turn) drives the choice>
  Risk: <what could still go wrong even with the recommended option>
  Pre-prototype diagnostic: <single test on Carnot's existing data
                              that would catch the recommended option's
                              failure before launching the full prototype>

UNRESOLVABLE UNCERTAINTY:
  <if there's an aspect of the action representation that genuinely
   cannot be diagnosed without running the prototype, name it
   explicitly so we don't pretend to know>
```

### Cross-validation reminder

Per `feedback_carnot_prediction_pattern.md`: prior Deep Think rounds
have qualitative survival claims well-calibrated, but specific
numerical prescriptions systematically wrong. This question is in
the representational-choice / failure-mode-analysis lane. If your
answer drifts toward parameter prescriptions (embedding dimension =
N, mass matrix entry = X, snap threshold = Y), please flag the
drift explicitly and provide the qualitative answer alongside.

Note: Q4 (verifier degradation) self-flagged 1 drift; Q5 (substrate
contamination) self-flagged 2 drifts; Q7 (HMC compatibility, asked
in parallel with this Q8) is in the same diagnostic-design lane.
The same self-discipline is welcome here.

This question pairs with Q7. Q7 asks "is HMC compatible with
Carnot's ∇E at all?" Q8 asks "given that HMC is compatible, what's
the right action-representation bridge for ARC-AGI-3?" If Q7
returns regime C (HMC inappropriate), Q8's analysis still applies
to whichever fallback sampler is chosen — the discrete-continuous
mismatch is a property of the problem, not of HMC specifically.

---

## End of prompt

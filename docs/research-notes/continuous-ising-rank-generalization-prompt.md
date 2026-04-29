# Deep Think Prompt — Continuous-to-Ising Generalization Error

**Status:** Ready to send. Closes a load-bearing gap in the Phase 2
transpiler theorem (REQ-PHASE2-006) — *expressive capacity* loss
under finite-bit-width Ising distillation, distinct from the Round-10
KL training-loss bound.

**Date drafted:** 2026-04-29

**Why this question:** Round-10 of the Zenil chain bounded the
*training-loss* gap (how close can the distilled Ising get to the
continuous EBM, given infinite samples and optimal training).
This question asks the *expressive-capacity* gap: even with infinite
samples and optimal training, what continuous EBMs simply cannot be
represented as Ising at finite bit-width? This is the Kolmogorov
$n$-width side of the transpiler theorem.

**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Setup

We have a hardware-accelerated EBM sampling system that maps a
continuous EBM $E_\theta: \Omega \to \mathbb{R}$ on a bounded domain
$\Omega \subset \mathbb{R}^d$ to a discrete Ising spin energy
$E_k: \{0, 1\}^{kd} \to \mathbb{R}$ via a Gray-code visible-spin
encoder at bit-width $k$ per dimension.

A prior derivation established the **training-loss bound**:

$$
\mathrm{KL}(\mu_P \| \tilde \mu_P^{(k)}) \leq \frac{d L^2}{24 T^2} \cdot 2^{-2k}
$$

where $\mu_P \propto \exp(-E_\theta/T)$ is the continuous Boltzmann
distribution, $\tilde \mu_P^{(k)}$ is the lifted distilled distribution,
$L = \|\nabla E_\theta\|_\infty$, and $T$ is temperature. This bound
is *dimension-independent in rate*: only the prefactor scales with
$d$, the rate $2^{-2k}$ is intrinsic.

The remaining question is the **expressive-capacity gap**: even with
infinite samples and optimal training, what continuous EBMs *cannot
be represented* by *any* Ising energy at bit-width $k$?

This is the Kolmogorov $n$-width / approximation-theory side of the
transpiler theorem, distinct from the training-loss side.

Use $L^2(\mu_P)$ throughout; make explicit any
$\mathrm{KL} \leftrightarrow L^2(\mu_P)$ conversions.

### Question 1: Capacity gap as a function of smoothness class

Define the function class
$\mathcal{F}_{\text{cont}}^{(s, L)} = \{E: \Omega \to \mathbb{R} \mid E \in C^s, \|\nabla E\|_\infty \leq L\}$
(Hölder-smooth with bounded gradient). Define
$\mathcal{F}_{\text{Ising}}^{(k)} = \{$Ising energies on $\{0,1\}^{kd}$ with arbitrary couplings$\}$.

For $E \in \mathcal{F}_{\text{cont}}^{(s, L)}$, define the
**capacity gap** as

$$
\Delta_{\text{cap}}(E, k) = \inf_{E' \in \mathcal{F}_{\text{Ising}}^{(k)}} \|E - \text{Lift}(E')\|_{L^2(\mu_P)}
$$

where $\text{Lift}$ embeds the discrete Ising energy back into
$L^2(\Omega)$ via Gray-code-cell-center evaluation.

(a) Derive the worst-case bound
$\sup_{E \in \mathcal{F}_{\text{cont}}^{(s, L)}} \Delta_{\text{cap}}(E, k) \leq C(d, L, s, T) \cdot 2^{-rk}$
for some rate $r > 0$.

(b) Compare to the Round-10 KL training-loss rate ($r = 2$). Is the
capacity-gap rate $r_{\text{cap}}$ *equal to*, *better than*, or
*worse than* the KL rate? Specifically: under what conditions does
$r_{\text{cap}} > r_{\text{KL}}$ (i.e., capacity is preserved
*better* than training, indicating a slack budget) versus
$r_{\text{cap}} < r_{\text{KL}}$ (capacity is the binding constraint)?

(c) For higher-smoothness classes ($s \geq 2$), does the rate
improve to $2^{-(s+1)k/d}$ (Smolyak-style) or remain at $2^{-2k}$
(the Round-10 KL rate)?

### Question 2: Pathological function classes that resist distillation

Some continuous EBMs cannot be approximated by *any* finite-bit-width
Ising representation:

(a) Identify the broadest function class $\mathcal{F}_{\text{patho}}$
such that $\sup_{E \in \mathcal{F}_{\text{patho}}} \Delta_{\text{cap}}(E, k) > c$
for all finite $k$ (i.e., the capacity gap is bounded below by a
positive constant regardless of bit-width).

(b) **Explicit construction.** Give a concrete example (preferably
relevant to Carnot's setting) of a continuous EBM in $\mathcal{F}_{\text{patho}}$.
Is the verifier energy $E$ for, e.g., the Z3-satisfaction-via-logical-
formula class in $\mathcal{F}_{\text{patho}}$? (Z3 verifiers contain
non-smooth indicator functions of formula satisfaction.)

(c) **Workaround for non-smooth verifiers.** If standard Carnot
verifiers (Z3 SMT, type checking) are in $\mathcal{F}_{\text{patho}}$,
how should we extend the transpiler? Options:
  - Smoothing: replace indicator with $\tanh$-bumped relaxation.
  - Subdomain decomposition: piecewise-constant on Gray-code cells
    each labelled by formula satisfaction.
  - Hybrid: keep verifier evaluation in software, only distill the
    smooth EBM core.

### Question 3: Inheritance of structural properties

Carnot's continuous EBMs have specific structural properties that
matter for downstream use:

(a) **Mode preservation.** If $E_\theta$ has $M$ distinct local
   minima (modes), how many are preserved in the distilled Ising
   $E_k$? Does the count increase, decrease, or stay constant as
   $k$ grows? Provide explicit lower/upper bounds.

(b) **Energy-gap preservation.** The energy gap between the
   ground-state mode and the second mode is the dominant scale for
   sampling difficulty (related to MCMC mixing time via Kramers).
   What's the relationship between the continuous gap $\Delta_{\text{cont}}$
   and the distilled gap $\Delta_k$? Specifically: is
   $|\Delta_{\text{cont}} - \Delta_k| \leq C \cdot 2^{-rk}$?

(c) **Convexity and Lipschitz preservation.** If $E_\theta$ is
   convex (or Lipschitz with constant $L$), is $E_k$? Or does the
   discretisation introduce non-convexity / increased Lipschitz?

### Question 4: Carnot-specific deployment recommendation

Synthesise into deployment guidance:

(a) For Carnot's typical EBM regime ($d \in [10, 100]$, $L \in [1, 10]$,
   $s = 2$, $T \approx 1$), what's the *minimum bit-width*
   $k_{\text{cap}}^{\text{min}}$ such that
   $\Delta_{\text{cap}} \leq 0.01 \cdot \|E_\theta\|_{L^2}$
   (1% capacity loss)?

(b) Is the binding constraint the KL training loss (Round-10) or
   the capacity gap (this round)?
$$
k^* = \max(k_{\text{KL}}^{\text{min}}, k_{\text{cap}}^{\text{min}})
$$
Which dominates for Carnot's typical case?

(c) **For non-smooth verifiers** (Z3, Boolean checks): explicit
   protocol for how Carnot should handle them. Smooth approximation
   first? Hybrid software/hardware execution? Direct-on-FPGA?

### Final integration: completing REQ-PHASE2-006

The Phase 2 transpiler theorem currently states:

> "The Gray-code visible-spin encoder reduces $\tau_{\text{int}}$ by
>  factor $\Gamma \approx \exp(\kappa(2^{k-1} - 1)/T)$ vs standard
>  binary encoding."

Augment this with the capacity statement:

> "Furthermore, for $E \in C^s(\Omega)$ with $\|\nabla E\|_\infty \leq L$,
>  the distilled Ising energy $E_k$ approximates $E$ in $L^2(\mu_P)$
>  with rate $\Delta_{\text{cap}}(k) \leq h(d, L, s, T) \cdot 2^{-r_{\text{cap}}k}$."

Provide $h$ and $r_{\text{cap}}$ explicitly. Identify the Carnot
function class for which the augmented theorem is fully deployable
(i.e., where the bound is non-vacuous and the function class covers
typical Carnot energies).

If non-smooth verifiers (Z3, type checks) lie outside the deployable
class, propose the architectural change required to bring them in
(e.g., smoothing parameters, hybrid execution, etc.).

---

## Internal cross-validation predictions (DO NOT PASTE)

### Q1 predictions

(a) Capacity gap: $\Delta_{\text{cap}} \leq C \cdot 2^{-2k}$ for
    $s \geq 2$, with $C = O(L \cdot \sqrt{d})$ — same rate as KL but
    different prefactor.
(b) For smooth functions, $r_{\text{cap}} = r_{\text{KL}} = 2$ —
    capacity and training loss both converge at same rate. KL is the
    binding constraint because its prefactor is smaller for typical
    parameters.
(c) Higher smoothness $s \geq 2$ improves: $r_{\text{cap}} = 2(s-1)$
    via Smolyak/sparse-grid approximation. So $s=2$ stays at rate 2,
    $s=3$ rate 4, etc.

### Q2 predictions

(a) Pathological class: functions with infinite Lipschitz constant
    (e.g., indicators), unbounded variation, fractal energy
    landscapes (Hausdorff dimension > $d$).
(b) Z3 verifiers ARE in $\mathcal{F}_{\text{patho}}$ for two reasons:
    (i) the satisfaction predicate is a Heaviside step (infinite
    Lipschitz at the boundary); (ii) the formula structure has
    discrete combinatorial complexity that doesn't correspond to
    continuous neighborhoods.
(c) Smoothing via $\tanh$ relaxation works in principle but
    reintroduces the verifier accuracy paradox (Round-12) — perfect
    smoothing → 0 information. Hybrid execution (verifier in
    software, EBM core on hardware) is the right answer.

### Q3 predictions

(a) Mode count: Ising $E_k$ has up to $2^{kd}$ distinct energy values,
    so $\geq M$ modes preserved if $2^{kd} \geq M$. For typical
    Carnot ($d=10$, $k=8$): $2^{80}$ available — modes preserved
    trivially.
(b) Energy gap: bound
    $|\Delta_{\text{cont}} - \Delta_k| \leq O(C \cdot 2^{-rk})$ with
    same rate $r$ as Q1.
(c) Convexity: distillation introduces non-convexity at cell
    boundaries (piecewise-constant has discrete jumps). Lipschitz:
    distilled $L_k \leq 2L$ (factor of 2 from cell-boundary
    discretisation).

### Q4 predictions

(a) For typical Carnot ($d=100$, $L=10$, $s=2$, $T=1$,
    $\varepsilon_{\text{cap}} = 0.01$):
    $k_{\text{cap}}^{\text{min}} = 8$ — same as Round-10's
    $k_{\text{KL}}^{\text{min}}$. Both rate-2, same prefactor up to
    constants.
(b) KL is binding (slightly smaller prefactor); capacity is
    near-binding. They scale together.
(c) Non-smooth verifiers: need hybrid execution. The verifier-energy
    side stays in software (CPU); only the smooth-EBM core distills
    to FPGA Ising.

### Final integration prediction

Augmented REQ-PHASE2-006:
- Smooth EBM core: deployable on FPGA at $k=8$ for 1% capacity loss.
- Z3-style verifiers: kept in software (hybrid execution).
- The Round-10 KL bound is the binding constraint; capacity gap is
  no worse and typically smaller.

This completes the Phase 2 transpiler theorem with both training-loss
and expressive-capacity guarantees.

## Action plan after response

1. **If Q1 confirms $r_{\text{cap}} = 2$ for $s=2$:**
   - REQ-PHASE2-006 augmented with capacity-gap statement.
   - Same $k_{\text{HW}}$ requirements as Round-10.

2. **If Q2 confirms Z3 is pathological:**
   - Update change proposal `zenil-grounded-self-distillation-deployable-stack.md`
     Exp G (autoencoder bottleneck) to include explicit
     hybrid-execution architecture.
   - Carnot's deployment story shifts: FPGA accelerates the smooth
     EBM core, CPU/dedicated hardware handles Boolean verifiers.

3. **If Q3 confirms mode preservation at $2^{kd} \gg M$:**
   - Multi-modal landscapes are safe; no architectural concern.

4. **Position-paper impact:**
   - 13th publishable contribution: complete Phase 2 transpiler
     theorem (training-loss + capacity).
   - Strategic: validates the FPGA acceleration roadmap doesn't
     hit a "fundamental" obstacle on smooth EBMs; pathological
     verifiers handled via known architectural patterns.

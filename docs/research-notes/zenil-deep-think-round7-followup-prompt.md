# Deep Think Round-7 Followup Prompt

**Status:** Ready to send. Builds on Round-6 validation results
(`zenil-deep-think-validation-results.md`). Asks Deep Think for two
deployment-blocking refinements of the orthogonality-stall theory.

**Date drafted:** 2026-04-28
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)` (between the two `---`
separators). Skip everything else — those are internal references.

---

## Prompt to send (verbatim)

### Background

We are deploying a verifier-filtered self-distillation loop for an
energy-based-model (EBM) framework. From a prior independent analysis,
the following are now established (use these as given premises;
re-derive only if you find an error):

1. **Effective grounding rate.** In Zenil's recursive-self-training
   formalism $\mu_{t+1} = (1-\alpha_t)\mu_t + \alpha_t \mu_P + \xi_t$,
   replacing the abstract $\alpha_t$ with a verifier-filter
   $w(x) \propto \exp(-E(x)/T)$ yields, in the high-T $L^2(\mu_P)$
   limit, the effective rate
   $\alpha_t^{\text{eff}} = \Phi/(T \delta_t)$ where:
   $$\Phi = \frac{1}{\delta_t}\,\mathrm{Cov}_{Q_t}\!\left(\frac{Q_t}{\mu_P},\,E\right) = \mathrm{Cov}_{\mu_P}(g_t, E)$$
   with $g_t = (Q_t - \mu_P)/(\delta_t \mu_P)$ the normalised residual
   error profile, and $\delta_t = \|Q_t - \mu_P\|_{L^2(\mu_P)}$.

2. **Stochastic contractivity bound.** Under MCMC sampling
   (PT-PCD/Gibbs) with variance scaling $\sigma^2 \tau_{\text{int}}/N_t$,
   strict contractivity $\mathbb{E}[\delta_{t+1}^2] < \delta_t^2$
   requires
   $N_t > \sigma^2 T \tau_{\text{int}} / (2 \Phi \delta_t).$

3. **Constant-T schedule.** Holding $T = T_{\text{floor}} \gtrsim T_{\text{crit}}$
   gives linear convergence $\Delta\delta_t \approx -\Phi/T_{\text{floor}}$
   per round, terminating at the structural noise floor
   $\mathcal{O}(\varepsilon)$ in finite time, with PT/replica exchange
   bridging $\tau_{\text{int}}$ blowup at phase transitions.

4. **The orthogonality stall.** Because $\Phi = \mathrm{Cov}_{\mu_P}(g_t, E)$,
   the verifier provides restorative force *only* along eigendirections
   where $g_t$ correlates with $E$. Each round eliminates the
   "$E$-aligned" component of $g_t$, so $g_t$ rotates into the
   null-space of $E$ and $\Phi \to 0$ even when $\delta_t$ is still
   large. Below the verifier's noise bound $\varepsilon$, the
   restorative step collapses; the loop stalls at an irreducible
   plateau $\delta_{\text{plateau}}$ that *no amount of $N_t$ can defeat*.

Use $L^2(\mu_P)$ norm throughout. Make explicit any
$\mathrm{KL} \leftrightarrow L^2(\mu_P)$ conversions used.

### Question 1: Quantify the orthogonality plateau

Given:
- A target distribution $\mu_P$ with score $\nabla \log \mu_P$.
- A verifier energy $E$ whose level sets define the constraint manifold.
- An initial residual error profile $g_0$ (decomposable into
  components aligned and orthogonal to $E$ in $L^2(\mu_P)$).

Derive a **closed-form expression or tight bound** for the
orthogonality plateau $\delta_{\text{plateau}}$ as a function of:

(a) The spectrum of $E$ (eigendecomposition of the linear operator
    $f \mapsto E \cdot f - \langle E \cdot f, \mathbf{1}\rangle_{\mu_P}$).
(b) The initial error's projection coefficients onto $E$'s eigenspaces
    (specifically the orthogonal complement projection
    $\Pi_{E^\perp} g_0$).
(c) The verifier accuracy $\varepsilon$ and noise floor.

Specifically: if $g_0$ has spectral decomposition
$g_0 = \sum_i c_i \phi_i$ in the eigenbasis of $E$'s covariance operator,
how does the residual evolve? At what rate does the $E$-aligned
component $\Pi_E g_t$ decay vs. the orthogonal component
$\Pi_{E^\perp} g_t$? When does the orthogonal mass dominate
$\widehat\Phi$ enough to drop it below $\varepsilon$?

Provide the explicit formula
$\delta_{\text{plateau}} = h(\|\Pi_{E^\perp} g_0\|, \varepsilon, \text{spectral gap of } E, ...)$
with all constants identified.

### Question 2: Multi-verifier coverage and rotation theory

To escape the orthogonality stall, propose using a finite set of
verifiers $\{E_1, ..., E_k\}$ whose union covers the residual-error
space. For Carnot's setting, each $E_i$ is a constraint-derived
energy (e.g., $E_1$ = type-checking constraints, $E_2$ = formal
properties, $E_3$ = Z3 satisfiability witnesses).

Derive:

(a) **Coverage condition.** What is the precise spectral condition
    on $\{E_1, ..., E_k\}$ (in $L^2(\mu_P)$) that guarantees the
    union of their non-null-spaces is dense in the residual-error
    space? Stated as: $\bigcap_i \mathrm{null}(E_i^*) = \{0\}$ in
    $L^2(\mu_P)$, where $E_i^*$ is the centred multiplication
    operator. Is this sufficient, or do we need a quantitative
    "minimum overlap angle" condition?

(b) **Optimal rotation cadence.** In a self-distillation loop,
    should the verifier rotate every round, or only when
    $\widehat\Phi_i$ falls below threshold? Derive the optimal
    policy. Specifically: under the corrected
    $\Phi = \mathrm{Cov}_{\mu_P}(g_t, E)$, does pre-emptive rotation
    (before stall) help, or does it just slow convergence on the
    currently-active eigendirections?

(c) **Ensemble combine vs. rotation.** Two architectures are possible:
    - **Rotation:** at round $t$, pick $E_{i^*}$ where $i^* = \arg\max_i \widehat\Phi_i$
      and use only that.
    - **Ensemble combine:** at round $t$, combine $w(x) = \prod_i \exp(-E_i(x)/T_i)$,
      effectively summing the energies.

    For each architecture, derive the effective $\Phi^{\text{eff}}$
    and the corresponding $N_t$ bound. Which dominates asymptotically?
    Is there a regime (e.g., when verifier energies are highly
    correlated vs. orthogonal) where one strictly beats the other?

(d) **Minimum number of verifiers.** Given a target $\delta_{\text{plateau}}^* < \delta_{\text{plateau}}^{\text{single}}$,
    what's the minimum $k$ for the rotation/ensemble architecture
    to achieve it? Express in terms of the residual-error space's
    intrinsic dimension and the $E_i$'s spectral coverage.

### Question 3: Distinguishing orthogonality from adversarial gaming

In deployment, a verifier-filtered self-distillation loop will
exhibit $\widehat\Phi \to 0$ for at least two distinct reasons:

- **Orthogonality stall (passive geometric):** residual $g_t$
  rotates into $E$'s null space because the model legitimately
  improved — the remaining errors are simply *not measurable* by $E$.
- **Adversarial gaming (active optimization):** the model learns to
  produce outputs that minimise $E$ without actually approaching
  $\mu_P$ — exploiting blind spots / Goodhart's Law on the verifier.

Both produce $\widehat\Phi \to 0$. Mitigations differ:
- Orthogonality → rotate to a different verifier (Q2).
- Gaming → the model is now adversarial; rotating to a *different*
  $E_j$ won't help if the model can also game $E_j$. Need to detect
  gaming, halt, retrain, or add an out-of-distribution detector.

Derive a **discriminating statistical test** that separates the two
regimes from observable quantities (samples, $\widehat\Phi$ history,
held-out evaluation):

(a) Specifically, characterise how the *higher-order statistics* of
    $(g_t, E)$ differ between the two regimes. E.g., does
    $\mathrm{Cov}_{\mu_P}(g_t, E^2)$ vanish in orthogonality but not
    in gaming? Or does the residual variance $\mathrm{Var}_{\mu_P}(g_t)$
    behave differently?

(b) Provide an explicit hypothesis test:
    $H_0$: orthogonality stall (legitimate convergence to plateau).
    $H_1$: adversarial gaming (illegitimate stall while $\delta_t$
    is artificially low or high).

    What's the test statistic, its null distribution, and the power
    against alternative-class gaming attacks?

(c) For Carnot's specific setting where $E$ is constraint-derived
    (Z3 witnesses, formal property checks), is there a structural
    *protection* against gaming that doesn't apply to learned
    verifiers (e.g., process reward models)? Or is gaming equally
    possible in both regimes?

### Final integration

Are answers to Q1, Q2, Q3 mutually consistent? Specifically: does
the discriminating test in Q3 have power against the gaming attack
that *exactly mimics* the orthogonality stall pattern from Q1? If
yes, that's the deployable safety net. If no, identify the
weakest-link assumption that an adversarial trainer could exploit.

---

## Internal cross-validation predictions (DO NOT PASTE)

For comparison after Deep Think replies:

### Q1 prediction
- $\delta_{\text{plateau}} = \|\Pi_{E^\perp} g_0\| \cdot \delta_0 / (1 - \rho)$
  for some contraction rate $\rho$ depending on spectral gap.
- Proportional to the orthogonal-component norm at $t=0$; verifier-
  aligned errors decay geometrically; orthogonal errors don't decay
  at all. Plateau is the orthogonal residual.
- $\delta_{\text{plateau}} \to 0$ when $E$ has full rank in the
  residual error space (no null directions).

### Q2 prediction
- (a) Coverage: $\bigcap_i \mathrm{null}(\mathrm{Cov}_{\mu_P}(\cdot, E_i)) = \{0\}$
  is sufficient when verifiers are linearly independent.
- (b) Pre-emptive rotation slows convergence; rotate only on stall.
  Each round on a single $E$ provides linear progress on its aligned
  subspace; rotating wastes that progress.
- (c) Ensemble combine effective $\Phi^{\text{ens}} = \sum_i \Phi_i$
  if uncorrelated; dominates asymptotically. Rotation $\Phi^{\text{rot}} = \max_i \Phi_i$
  is robust to badly-correlated $E$'s. Ensemble is more efficient
  when known-uncorrelated; rotation is safer when correlations are
  unknown.
- (d) Minimum $k$ = intrinsic dimension of residual error space
  (covering basis).

### Q3 prediction
- Orthogonality stall preserves $\delta_t \approx \delta_{\text{plateau}}$
  with consistent $\widehat\delta$ across held-out evaluations.
- Adversarial gaming may show $\widehat\delta_t$ low on the verifier-
  measured space but high on held-out evaluation (the gap is the
  Goodhart signature).
- Discriminating test: estimate $\delta_t$ on a *held-out* data
  source (different evaluator). If orthogonality, $\widehat\delta_{\text{held}} \approx \delta_{\text{plateau}}$.
  If gaming, $\widehat\delta_{\text{held}}$ is much larger.
- Carnot's constraint-derived energy is structurally protected: Z3
  satisfaction is verifiable, not learned, so model can't game the
  oracle. Learned PRMs can be gamed.

## Carnot-side action plan after Round-7

If Deep Think confirms Q1 and Q2, this directly informs the
implementation of Exp E (orthogonality-stall mitigation) in the
change proposal. Specifically:

1. The δ_plateau formula gives Phase 2 hardware sizing — when does
   FPGA-acceleration stop helping?
2. The minimum $k$ tells us how many constraint sets to ship in the
   multi-verifier rotator.
3. The Q3 discriminating test ships as an in-loop safety detector.

If Deep Think identifies a fundamental obstacle to *any* of these,
that's a publishable negative result and reshapes Phase 3
architecture.

If Deep Think confirms Carnot's structural protection against gaming
in Q3(c), that's a load-bearing argument for the constraint-derived-
energy approach over learned PRMs — should be added to the
decentralization-respecting design rules in CLAUDE.md.

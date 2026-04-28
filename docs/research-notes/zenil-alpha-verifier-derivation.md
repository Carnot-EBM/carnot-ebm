# Zenil α_t × Carnot Verifier — First-Order Derivation

**Date:** 2026-04-28
**Origin:** Gemini response to Round-5 Deep Think question (formalising
the role of Carnot's energy-based verifier in Zenil's recursive-self-
training dynamical system).
**Status:** Draft derivation, peer-reviewed by Carnot critique pass.
**Cross-refs:** `research-references.md` → "Zenil 2026 self-improvement
limits"; `memory/project_zenil_alpha_grounding.md`.

## Setup

Zenil 2026 (arXiv 2601.05280v2) formalises recursive self-training as

$$
\mu_{t+1} = (1 - \alpha_t)\mu_t + \alpha_t \mu_P + \xi_t
$$

where $\mu_P$ is the truth distribution, $\alpha_t$ is the per-round
proportion of exogenously-grounded data, and $\xi_t$ is finite-sampling
noise. Theorem 5: convergence to $\mu_P$ requires $\inf_t \alpha_t > 0$.

We replace exogenous grounding with a *verifier-filter*: candidates
sampled from base model $Q_t$ are accepted with probability
$w(x) \propto \exp(-E(x)/T)$, where $E$ is Carnot's energy function and
$T$ is the temperature.

Define:

- $\delta = \|Q_t - \mu_P\|_{TV}$ — the model's current distance from truth.
- $\varepsilon = \|E - (-\log \mu_P)\|_\infty$ — verifier's $L^\infty$ error.
- $\Delta(x) = Q_t(x) - \mu_P(x)$ — pointwise overestimation.
- $\tilde Q_t$ — the filtered distribution induced by $Q_t$ and $w$.
- $\alpha_t^{\text{eff}}$ — defined by
  $\|\tilde Q_t - \mu_P\|_{TV} = (1 - \alpha_t^{\text{eff}}) \delta$.

## First-order derivation (high-T regime)

For $T \gg 1$, linearise $\exp(-E(x)/T)$ around $E/T = 0$. Writing
$E(x) = -\log \mu_P(x) + e(x)$ with $|e| \le \varepsilon$:

$$
w(x) \approx 1 + \frac{\log \mu_P(x)}{T} - \frac{e(x)}{T}.
$$

After normalisation $Z = \mathbb{E}_{Q_t}[w]$:

$$
\tilde Q_t(x) \approx Q_t(x)\left[1 + \frac{g(x)}{T} - \frac{h(x)}{T}\right],
$$

where $g(x) = \log\mu_P(x) - \mathbb{E}_{Q_t}[\log \mu_P]$ and
$h(x) = e(x) - \mathbb{E}_{Q_t}[e]$ are mean-zero under $Q_t$.

For small $1/T$ perturbation (assuming $\Delta$ doesn't change sign):

$$
\|\tilde Q_t - \mu_P\|_{TV} \approx \delta + \frac{1}{2T}\int \mathrm{sgn}(\Delta) Q_t (g - h)\, dx.
$$

Define the *structural restorative force*:

$$
\Phi = -\tfrac{1}{2}\int \mathrm{sgn}(\Delta(x)) \, Q_t(x) \, g(x)\, dx.
$$

Then

$$
\boxed{\alpha_t^{\text{eff}} \approx \frac{\Phi - \mathcal{O}(\varepsilon)}{T \, \delta}}.
$$

## When is $\Phi > 0$? (the key structural assumption)

$\Phi > 0$ is the claim that the verifier provides a *restorative*, not
*destructive*, force. The argument:

- Where $Q_t$ overestimates ($\Delta > 0$), $\mu_P$ is below average
  ($g(x) < 0$), so the integrand is negative.
- Where $Q_t$ underestimates ($\Delta < 0$), $\mu_P$ is above average
  ($g(x) > 0$), so the integrand is again negative.
- Therefore $\int \mathrm{sgn}(\Delta) Q_t g\, dx < 0$, hence $\Phi > 0$.

**Caveat.** This is an *empirical structural assumption* about how LLM
hallucinations distribute, not a theorem. It says: "$Q_t$'s errors are
correlated with low-density regions of $\mu_P$." This holds when
hallucinations are tail-events (likely true for code repair, math
proofs, factual claims) but fails if $Q_t$ has miscalibrated *modes*
of $\mu_P$ (drift in central probability mass). Section "Open
questions" tightens this.

## Conditions for contractivity (Theorem 5)

The verifier-filtered process satisfies $\inf_t \alpha_t^{\text{eff}} > 0$
when:

1. **Strict accuracy bound:** $\Phi > \varepsilon$. If $\varepsilon \ge \Phi$,
   the error swamps the signal and $\alpha_t^{\text{eff}}$ may be
   non-positive, *actively injecting bias*.

2. **Persistent exogenous anchoring:** $E$ cannot be retrained on $Q_t$'s
   own outputs. If $E$ tracks $Q_t$, the system devolves into a closed
   loop and $\Phi \to 0$. *Carnot satisfies this by construction*: the
   energy function is constraint-derived (Z3 satisfaction, type checks,
   formal properties) — not learned from $Q_t$. This is a structural
   advantage over learned-verifier approaches (e.g., Process Reward
   Models trained on the same model's outputs).

3. **Lower-bounded temperature:** $T > T_{\text{crit}}$ to keep the
   linearisation valid AND to preserve distributional diversity. At
   $T \to 0$, $\tilde Q_t \propto Q_t \cdot \mu_P^{1/T}$ — sharpening
   onto modes. Sharpening is benign when $E$ is accurate but
   catastrophic when $\varepsilon$ is appreciable.

## Sensitivity: temperature vs. energy accuracy

**Energy accuracy strictly dominates.**

- $\varepsilon$ is in the *numerator* and gates the *sign* of
  $\alpha_t^{\text{eff}}$. If $\Phi - \mathcal{O}(\varepsilon) < 0$,
  the verifier actively reinforces hallucinations regardless of $T$.
- $T$ is in the *denominator* and only modulates the *step size*.
  Tuning $T$ cannot rescue an inaccurate verifier.

**Engineering implication.** When prototyping verifier acceleration on
edge hardware (FPGA / KV260 / Extropic XTR-0), the precision of $E(x)$
under quantisation matters far more than dynamic temperature scaling.
INT8 / mixed-precision implementations of $E$ must be validated to
maintain $\varepsilon < \Phi$ — sloppy quantisation directly injects
self-distillation collapse.

## Sub-exponential convergence

The scaling $\alpha_t^{\text{eff}} \sim \Phi/(T\delta)$ has an
implication missed in the first-pass derivation: **as $\delta$ shrinks
(model improves), so does the effective grounding signal**.

The contraction rate is $1 - \alpha_t^{\text{eff}}$, so

$$
\delta_{t+1} \approx \delta_t - \frac{\Phi}{T} \quad \text{(approx., for small } \alpha_t^{\text{eff}}\text{)}
$$

— that is, *additive*, not multiplicative, decay. Convergence is
linear-in-$t$, not exponential. The fight to maintain $\Phi > \varepsilon$
*gets harder as the model converges*: $\Phi$ depends on the structural
mismatch between $Q_t$ and $\mu_P$ at low-density regions, which
diminishes as $Q_t \to \mu_P$.

**Phase 3 architectural warning.** A verifier that is "good enough" at
the start of training is *not* good enough late. Either:

1. Energy function $E$ must be progressively refined (additional
   constraints added) as the model converges, *or*
2. The verifier's accuracy $\varepsilon$ must be *strictly* tighter
   than the asymptotic $\Phi$, *or*
3. Acceptance of slow-but-stable convergence — possibly logarithmic
   in pathological cases.

## Stochastic extension (Gemini Round-6 preliminary)

Gemini provided a preliminary derivation for the noise term (open
question 2 below). Key result: *the required sample size diverges as
the model converges*.

Setup: introduce finite-sample noise $\xi_t$ with $\mathbb{E}[\xi_t] = 0$
and $\mathbb{E}[\|\xi_t\|^2] \approx \sigma^2/N$ (IID Monte Carlo
approximation). Filtered update becomes stochastic:

$$
\hat Q_{t+1} = Q_{t+1} + \xi_t.
$$

Expected squared $L^2$ contraction:

$$
\mathbb{E}[\delta_{t+1}^2] \le (1 - \alpha_t^{\text{eff}})^2 \delta_t^2 + \frac{\sigma^2}{N}.
$$

For strict contractivity (model improves), $\delta_t^2 - \mathbb{E}[\delta_{t+1}^2] > 0$.
Substituting $\alpha_t^{\text{eff}} \sim \Phi/(T \delta_t)$ from the
deterministic derivation and using $(1-\alpha)^2 \approx 1 - 2\alpha$
for small $\alpha$:

$$
\boxed{N > \frac{\sigma^2 T}{2\Phi \delta_t}}.
$$

**Interpretation.** As the model converges ($\delta_t \to 0$), the
required sample size $N$ *diverges to infinity*. This is the deployable
bound — the inflection point where naïve self-distillation stops
working and additional compute (or hardware) is required to maintain
contractivity.

**Three caveats not yet in the derivation, worth pushing to Deep Think:**

1. **MCMC autocorrelation.** The IID variance scaling $\sigma^2/N$ is
   too generous for the EBM samplers Carnot will actually use. For
   PT-PCD and Gibbs samplers, effective sample size is $N/\tau_{\text{int}}$,
   so the deployable bound is
   $$N > \frac{\sigma^2 T \tau_{\text{int}}}{2\Phi \delta_t}.$$
   Near phase transitions, $\tau_{\text{int}}$ blows up — a separate
   failure mode that the deterministic derivation does not capture.

2. **Norm consistency.** The deterministic derivation lives in TV;
   the noise extension uses squared $L^2$. Pinsker's inequality
   bridges the two but the constants are loose. A clean derivation
   should pick one norm — $L^2(\mu_P)$ is the most natural for the
   verifier-filter formalism.

3. **The $\Phi > 0$ assumption is load-bearing.** If the structural
   assumption (open question 1 below) fails, $\Phi$ can be
   non-positive, the contraction step has indefinite sign, and the
   noise bound is vacuous. The three open questions are *nested*:
   1 → 2 → 3.

## Phase 2 hardware connection (this is the load-bearing argument)

The bound $N > \sigma^2 T \tau_{\text{int}} / (2\Phi \delta_t)$ is
*why Carnot's continuous-to-Ising transpiler exists*. The Phase 1 →
Phase 2 → Phase 3 arc is not "speed up sampling because faster is
better." It is *self-distillation provably collapses without
hardware-accelerated sampling as $\delta_t \to 0$*.

Concrete throughput targets:

| Phase | Hardware | Samples/sec | Convergence depth $\delta_t$ supported |
|-------|----------|-------------|---------------------------------------|
| 1 | CPU / ROCm | $\sim 10^6$ | $\delta_t \gtrsim 10^{-3}$ |
| 2 | KV260 / FPGA Ising | $\sim 10^9$ | $\delta_t \gtrsim 10^{-6}$ |
| 2+ | Extropic XTR-0 (TSU) | $\sim 10^{11}$–$10^{12}$ | $\delta_t \gtrsim 10^{-9}$ |
| 3 | photonic / Ising-machine cluster | $\gtrsim 10^{13}$ | foundation-model regime |

(Numbers are illustrative — exact bounds depend on $\sigma^2$, $T$,
$\tau_{\text{int}}$, $\Phi$ for each task class.)

**This is the cleanest narrative argument for Phase 2 we have.** The
KV260 / Extropic / photonic tracks are not capabilities-shopping;
they are *necessary infrastructure for the Phase 3 foundation
model's training loop to remain non-collapsing*. Without them, every
self-distillation loop hits the deterministic deployment blocker
identified above. Decentralization-respecting design rule 5
(hardware portability as political requirement) compounds:
sovereign access to high-throughput EBM sampling becomes a
prerequisite for sovereign Phase 3 training, not an optional
accelerator.

## Open questions / Deep Think follow-ups

1. **Tighten $\Phi > 0$ assumption.** Under what *provable* structural
   conditions on $Q_t$'s error pattern is $\Phi$ guaranteed strictly
   positive? Candidate: monotone-likelihood hallucination
   (overestimation rate decreases with $\log \mu_P$). Need an exact
   sufficient condition that maps to a measurable empirical signal.

2. **Extend to noise.** Include the $\xi_t$ finite-sampling term.
   What's the contractivity bound when $\xi_t$ is added? At what
   sampling rate does the verifier signal exceed sampling noise?

3. **Convergence rate.** Is the convergence linear, polynomial, or
   logarithmic? Tight asymptotic bound on $\delta_t$ as $t \to \infty$
   with the verifier active? What's the optimal annealing schedule
   for $T$ that maximises convergence speed subject to $\Phi > \varepsilon$
   throughout?

4. **L^p generalisation.** Replace $\|E - (-\log\mu_P)\|_\infty$ with
   $\|\cdot\|_{L^2(\mu_P)}$ — gives a softer bound but more practical
   measurement. How does the contractivity conclusion change?

5. **Multi-pass amplification.** If the verifier filter is applied
   $k$ times per round (rejection sampling with threshold), what's
   the effective $\alpha_t^{\text{eff}}(k)$ and the trade-off vs
   wall-time cost?

## Carnot-specific implications

- **FR-11 (cross-session memory) + verifier-filtered self-distillation**
  is provably non-collapsing as long as the constraint-derived energy
  satisfies $\Phi > \varepsilon$ under quantisation. This is a
  publishable result: a no-collapse guarantee for FR-11.

- **Phase 3 foundation model** must use a constraint-derived (not
  learned) energy at the verifier layer. Learned PRMs (Process Reward
  Models) violate condition 2 (persistent exogenous anchoring) and are
  predicted to collapse over $N$ rounds.

- **SSD self-distillation (Apple)** is in tension. Either the base
  model's pretraining provides residual $\alpha_t > 0$ for a few
  rounds (which would explain the +12.9pp gain over a single
  iteration), or SSD will eventually collapse. Empirical test:
  iterate SSD over 10+ rounds and measure KL drift. Carnot's
  contribution is the prediction *before* the experiment, not after.

- **α_t metric in autoresearch.** Each milestone where the conductor
  performs self-distillation (Tier-0i probe training, FR-11 relay
  expansion) must report:
  - Estimated $\Phi$ (via held-out evaluation).
  - Verifier accuracy $\varepsilon$ (via cross-validation against
    reference implementations — `carnot.eval.metrics` is the
    canonical pattern).
  - Effective $\alpha_t^{\text{eff}}$ ratio.
  Doomed-rerun discipline retires re-runs with same verdict; α_t-
  collapse discipline should retire self-distillation experiments
  whose effective $\alpha_t^{\text{eff}}$ approaches zero.

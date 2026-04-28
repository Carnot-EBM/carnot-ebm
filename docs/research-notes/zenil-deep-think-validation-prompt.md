# Deep Think Validation Prompt: Zenil-Verifier Bound (Gemini Sanity-Check)

**Status:** Ready to send. This document asks Deep Think to independently
validate, refute, or refine claims that a lesser Gemini model produced
during prompt iteration.

**Date:** 2026-04-28
**Carnot context:** This work is the mathematical foundation for the
Carnot EBM project's Phase 2 / Phase 3 self-distillation architecture.
A bundled change proposal is filed at
`openspec/change-proposals/zenil-grounded-self-distillation-deployable-stack.md`.

## How to use this file

Open a fresh Deep Think session. Paste the section labelled
"## Prompt to send (verbatim)" below. Do NOT paste this header or any
"Carnot pre-validation context" notes — Deep Think should reason
independently before being shown what Gemini claimed.

---

## Prompt to send (verbatim)

### Background and setup

We are analyzing the convergence of an energy-based-model (EBM)
self-distillation loop. The setup is Zenil's recursive-self-training
formalism (arXiv 2601.05280v2):

$$
\mu_{t+1} = (1 - \alpha_t)\mu_t + \alpha_t \mu_P + \xi_t
$$

where $\mu_P$ is the truth distribution, $\alpha_t$ is the per-round
proportion of exogenously-grounded data, and $\xi_t$ is finite-sample
noise. Zenil's Theorem 5: convergence to $\mu_P$ requires
$\inf_t \alpha_t > 0$.

We replace the abstract $\alpha_t$ with a **verifier-filter**:
candidates from base model $Q_t$ are accepted with probability
proportional to $\exp(-E(x)/T)$, where $E$ is an energy function
(constraint-derived in our setting — Z3 satisfaction, type checks,
formal-property violations) and $T$ is filter temperature. The
verifier-filtered distribution $\tilde Q_t$ defines an *effective*
$\alpha_t^{\text{eff}}$ via
$\|\tilde Q_t - \mu_P\|_? = (1 - \alpha_t^{\text{eff}}) \delta_t$,
where $\delta_t$ is the model's distance from truth.

Use $L^2(\mu_P)$ as the working norm throughout. Make explicit any
$\mathrm{KL} \leftrightarrow L^2(\mu_P)$ conversions; do not hide
constants in norm changes.

### What we have derived (deterministic case)

A first-order $L^2(\mu_P)$ derivation under high-T linearisation has
established:

$$
\alpha_t^{\text{eff}} \approx \frac{\Phi - \mathcal{O}(\varepsilon)}{T \, \delta_t}
$$

where:
- $\delta_t = \|Q_t - \mu_P\|_{L^2(\mu_P)}$
- $\varepsilon = \|E - (-\log \mu_P)\|_\infty$ is the verifier's
  $L^\infty$ accuracy.
- $\Phi$ is the **structural restorative force** — a functional of
  $Q_t$, $\mu_P$, and the verifier's energy $E$ that captures
  whether the verifier's filter pulls $Q_t$ *toward* $\mu_P$ or
  *away from* it.

Three contractivity conditions follow from this derivation:
1. $\Phi > \varepsilon$ (verifier accuracy strictly dominates step-size).
2. Persistent exogenous anchoring (E not learned from $Q_t$ — Carnot's
   constraint-derived energy satisfies this by construction).
3. $T > T_{\text{crit}}$ (Taylor validity + entropy preservation).

### Five questions

We obtained answers from a Gemini model. Please reason **independently**
through each question first, then compare your derivation to the Gemini
claim labelled "Prior claim to validate." Mark each prior claim as
*confirmed*, *refined*, or *refuted*, with explicit corrections where
they diverge.

---

#### Question 1: Structural sufficient condition for $\Phi > 0$

Define the exact, *measurable* condition on the error pattern
$(Q_t - \mu_P)$ that guarantees $\Phi > 0$ strictly in $L^2(\mu_P)$.

**Prior claim to validate (Gemini):**
"$\Phi$ is the spectral gap of the Langevin generator $\mathcal{L}_E$
associated with $E$:
$$
\Phi = \inf_{f: \|f\|_2 = 1} \langle f, \mathcal{L}_E f\rangle_{L^2(\mu_P)}.
$$
Sufficient condition: monotone-likelihood hallucination plus the
integral inequality
$$
\int B(x) \, \nabla \log \mu_P(x) \, d\mu_P(x) < 0,
$$
where $B(x) = \log Q_t(x) - \log \mu_P(x)$ is the verifier's bias —
i.e., the bias is anti-aligned with the score of truth. This ensures
the verifier suppresses modes where $\nabla \log \mu_P$ is misaligned
with the proposal gradient."

Does the spectral-gap reframing hold? Is the integral condition
necessary, sufficient, or both? What is the *tightest* sufficient
condition expressible in terms of measurable quantities of
$Q_t - \mu_P$?

---

#### Question 2: Stochastic MCMC contractivity bound

Move from the deterministic regime to a stochastic one with finite-
sample noise $\xi_t$. We rely on MCMC samplers (PT-PCD, Gibbs), so
variance scales as $\sigma^2 \tau_{\text{int}} / N_t$ where
$\tau_{\text{int}}$ is the integrated autocorrelation time and $N_t$
is the per-round sample budget.

Derive the expected contractivity condition
$\mathbb{E}[\delta_{t+1}^2] < \delta_t^2$ in $L^2(\mu_P)$. Prove the
critical sample-size bound
$N_t > f(\sigma^2, T, \tau_{\text{int}}, \Phi, \delta_t)$ at which
the verifier's grounding signal is overwhelmed by sampling noise.

**Prior claim to validate (Gemini):**
"The bound is $N_t > \sigma^2 \tau_{\text{int}} / (2 \alpha_t \Phi \delta_t^2)$,
quadratic in $1/\delta_t$. When the deterministic
$\alpha_t = \Phi / (T \delta_t)$ is substituted, this becomes
$N_t > \sigma^2 T \tau_{\text{int}} / (2 \Phi^2 \delta_t)$, linear
in $1/\delta_t$. Both forms are correct in their parameterisation."

Is the quadratic-vs-linear distinction artefactual or substantive?
Does the leading $1/2$ constant come from the squared-$L^2$
contraction expansion, or is there a missed sub-leading term?

What is the rescue at critical phase transitions where
$\tau_{\text{int}} \to \infty$? Replica exchange, parallel-tempering
ladder, or a fundamental obstacle?

---

#### Question 3: Optimal joint annealing schedule

Given that $N \to \infty$ as $\delta_t \to 0$ from question 2, and
$\tau_{\text{int}}$ may blow up at phase transitions, derive the
*optimal joint asymptotic schedule* for $T_t$ and $N_t$ as
$t \to \infty$.

Provide the deployable formula with **leading constants**, not just
$T_t \sim t^{-\alpha}$ scaling. We need numbers we can put in a
training-loop hyperparameter file.

**Prior claim to validate (Gemini):**
"The optimal joint schedule is
$$
T_t = T_0 \frac{\log t_0}{\log t}, \qquad N_t = N_0 \left(\frac{\log t}{\log t_0}\right)^2.
$$
The budget couples to the lattice spectral gap as
$N_t \approx \sigma^2 \tau_{\text{int}} / (\lambda_{\min} T_t)$.
Convergence is logarithmic: $\delta_t \sim 1 / \log t$ in the worst
case."

Confirm or refute the $1/\log t$ rates. What are the explicit
$T_0, N_0, t_0$ constants in terms of the problem parameters
$\Phi, \varepsilon, \sigma, \tau_{\text{int}}$? Is logarithmic
convergence actually optimal, or can a more aggressive schedule
(e.g., polynomial) achieve faster convergence under stronger
assumptions on $\Phi$?

---

#### Question 4: Gray-code transpiler $\tau_{\text{int}}$ improvement factor

Our hardware-acceleration target maps continuous variables to
Ising spin space using a Gray-code visible-spin encoder that
maintains Hamming distance 1 between adjacent quantization cells.
This eliminates artificial energy "cliffs" present in standard
binary encoding (where adjacent integers can differ by many bits,
requiring multi-spin-flip transitions that exponentially suppress
mixing).

Derive the exact mathematical relationship (improvement factor) between
this Gray-code locality and the reduction in $\tau_{\text{int}}$
compared to standard binary encoding at bit width $k$.

**Prior claim to validate (Gemini):**
"The improvement factor is
$$
\Gamma = \frac{\tau_{\text{int}}^{\text{binary}}}{\tau_{\text{int}}^{\text{gray}}} \approx \exp\left(\frac{\kappa}{T} \sum_{i=1}^{k}(2^i - 1)\right),
$$
where $\kappa$ is the underlying Hamiltonian's coupling strength.
For $k=8$: $\sum_{i=1}^{8}(2^i - 1) = 1004$, so
$\Gamma \approx \exp(1004 \kappa/T)$ — super-exponential
improvement in bit width."

Is the super-exponential factor correct? Does the constant 1004
include the right combinatorial counting of cliff-crossing paths,
or is there double-counting? Does this break down at extreme low T
(where the Gray-code's own genuine-barrier transitions also
suppress)?

---

#### Question 5 (deployable hyperparameter): Optimal PT acceptance rate

The standard Roberts-Rosenthal limit for parallel-tempering swap
acceptance is $\approx 0.23$ for IID-target Metropolis. For
verifier-filtered sampling — where the proposal already biases
toward "good" samples — does the optimal swap acceptance rate
shift?

**Prior claim to validate (Gemini):**
"For verifier-filtered sampling, target a $\approx 0.35$
acceptance rate. The acceptance-reward trade-off shifts toward
higher temperatures (more mode-hopping over local-mode
convergence)."

Is 0.35 correct? Provide the derivation, and ideally the
parametric dependence on $\Phi$, $T$, $\varepsilon$ — the optimum
likely depends on how aggressive the verifier filter is.

---

### Final integration check

Are claims 1–5 *mutually consistent* and *jointly deployable*? In
particular:

- Does the spectral-gap definition of $\Phi$ in (1) compose with
  the contractivity bound in (2)? Specifically: does the
  $\Phi$ in (2) refer to the same quantity as in (1), or are
  there two different $\Phi$'s?
- Does the annealing schedule in (3) preserve the
  $\Phi > \varepsilon$ condition needed for (1) and (2)?
- Does the Gray-code factor in (4) actually feed into (3)'s
  $N_t$ formula via reduced $\tau_{\text{int}}$? If so, what
  $N_t$ reduction does the empirical Gray-code factor produce
  in deployment?
- Is 0.35 PT acceptance from (5) consistent with the schedule
  from (3)?

If you find the prior claims fully consistent, please *also*
identify the single weakest link in the derivation — the place
most likely to fail under realistic assumptions.

---

## Carnot pre-validation context (DO NOT PASTE TO DEEP THINK — internal reference only)

The Round-6 Gemini response that this validation prompt audits is
captured in `docs/research-notes/zenil-deep-think-round6-prompt.md`.
The full mathematical derivation chain is in
`docs/research-notes/zenil-alpha-verifier-derivation.md`.

After Deep Think replies, our action plan is:

1. **For each of 5 questions:** record Deep Think's verdict
   (confirmed / refined / refuted) in the Round-6 file.
2. **If any claim is refuted:** roll back the corresponding shippable
   artifact in the change proposal
   `zenil-grounded-self-distillation-deployable-stack.md`. The
   Gray-code factor (Q4) is the one whose refutation would be most
   consequential — that's the publishable Phase 2 theorem.
3. **If all claims hold:** proceed with the change proposal as-is and
   plan to ship in milestone .81 or .82.
4. **If Deep Think identifies a weakest-link:** add an explicit caveat
   to the change proposal's "Risks" section so future implementers
   know where to be careful.

The Carnot project's Phase 2 hardware mandate
(`_bmad/architecture.md` → "Asymptotic Hardware Mandate") rests on
claims 2, 3, and 4 holding. If Deep Think refutes the
$N \sim 1/\delta_t$ scaling or the Gray-code super-exponential
improvement, the hardware mandate's mathematical justification
weakens, and we'd need to redo the architecture argument.

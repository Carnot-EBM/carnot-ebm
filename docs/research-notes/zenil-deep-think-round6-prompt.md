# Deep Think Round-6 Prompt: Stochastic Zenil-Verifier Bound

**Status:** Ready to send. Iterated through three rounds of Gemini ↔ Carnot
critique to airtight form.

**Date drafted:** 2026-04-28
**Cross-refs:** `docs/research-notes/zenil-alpha-verifier-derivation.md`,
`_bmad/architecture.md` → "Asymptotic Hardware Mandate"

## Prompt (verbatim, ready to send)

System/Context:

> We are rigorously analyzing the convergence of an EBM self-distillation
> loop (where a verifier filters generative samples). We target a hardware
> deployment mapping continuous energy landscapes to Ising/FPGA
> accelerators, meaning finite-sample scaling and exact mathematical
> bounds dictate physical architecture. All derivations must remain
> strictly within the $L^2(\mu_P)$ norm to ensure consistency between
> structural error and noise bounds. You must make explicit any
> $\mathrm{KL} \leftrightarrow L^2(\mu_P)$ conversions used; do not hide
> constants in norm changes.

> Please synthesize the following four constraints:

**1. The Structural Restorative Force ($\Phi$):**
For the effective learning rate $\alpha_t^{\text{eff}}$ to be strictly
positive, the structural restorative force $\Phi$ must be $> 0$.
Define the exact, measurable condition on the error pattern
$(Q_t - \mu_P)$ that guarantees $\Phi > 0$ strictly in $L^2(\mu_P)$.
(e.g., Is monotone-likelihood sufficient? What is the formal
structural test to ensure the verifier focuses on valid energy modes
rather than hallucinated ones?)

**2. The Stochastic MCMC Bound:**
Moving from the deterministic to the stochastic regime, introduce
finite-sample approximation noise $\xi_t$. Because we rely on MCMC
(PT-PCD/Gibbs), assume the variance scales as
$\sigma^2 \tau_{\text{int}} / N$, where $\tau_{\text{int}}$ is the
integrated autocorrelation time. Derive the expected contractivity
condition $\mathbb{E}[\delta_{t+1}^2] < \delta_t^2$ in $L^2(\mu_P)$.
Prove the critical sample size bound
$N_t > f(\sigma^2, T, \tau_{\text{int}}, \Phi, \delta_t)$ where the
verifier's grounding signal is overwhelmed by sampling noise.

**3. The Asymptotic Annealing Schedule:**
Given that the derived bound in #2 requires $N \to \infty$ as
$\delta_t \to 0$, and recognizing that $\tau_{\text{int}}$ blows up
near phase transitions in the energy landscape, derive the optimal,
joint asymptotic annealing schedule for temperature $T_t$ and
sample budget $N_t$ as $t \to \infty$. Do not just provide the
scaling law ($T_t \sim t^{-\alpha}$); provide the deployable
formula with the exact leading constants to guarantee convergence
without requiring infinite compute.

**4. Transpiler Mitigation of $\tau_{\text{int}}$:**
To mitigate MCMC mixing degradation near phase transitions, our
hardware transpiler maps continuous variables to an Ising space
using a Gray-code visible-spin encoder. This ensures adjacent
quantization cells maintain a Hamming distance of 1, eliminating
the artificial energy "cliffs" present in standard binary encoding.
Derive the exact mathematical relationship (the improvement factor)
between this Gray-code locality and the reduction in the integrated
autocorrelation time $\tau_{\text{int}}$ compared to standard binary
encoding.

## Pre-registered Carnot predictions

These are our best guesses *before* sending the prompt. After Deep
Think responds, we cross-validate to identify where our intuition was
right vs surprised. This is part of the literature-priority discipline
(see `feedback_literature_priority_discipline` memory).

### Prediction (1) — Φ > 0 condition

- **Sufficient (we believe):** monotone-likelihood hallucination —
  overestimation rate is non-increasing in $\log \mu_P(x)$.
- **Necessary AND sufficient (suspect):** the inner product
  $\langle \mathrm{sgn}(Q_t - \mu_P), \log \mu_P - \mathbb{E}_{Q_t}[\log \mu_P]\rangle_{L^2(\mu_P)} < 0$
  (negative correlation between hallucination and log-truth-density).
- **Estimable via:** bootstrap CI on the inner product computed on a
  held-out evaluation set; reject null $\Phi = 0$ at $\alpha=0.05$.

### Prediction (2) — Stochastic bound

- $N_t > \sigma^2 T_t \tau_{\text{int}} / (2 \Phi \delta_t)$ (with leading
  constant 1/2 from the squared-$L^2$ contraction).
- Phase-transition rescue: parallel tempering with replica spacing
  chosen so swap acceptance rate ≈ 0.23 (Gelman-Roberts-Gilks rule).
  Replica exchange exponentially reduces effective $\tau_{\text{int}}$
  via mode-swapping at high T.

### Prediction (3) — Asymptotic schedule

- $T_t \sim T_0 / \log(t)$ (cooling slow enough to maintain $\Phi > \varepsilon$).
- $N_t \sim \sigma^2 T_t \tau_{\text{int}} / (2\Phi \delta_t)$ — track
  the bound exactly, not over-budget.
- $\delta_t \sim 1 / \log(t)$ — convergence is logarithmic in $t$
  in the worst case (the "fight to maintain $\Phi > \varepsilon$"
  dominates the rate).
- For pathological landscapes ($\tau_{\text{int}}$ blows up), even
  worse — possibly $\delta_t \sim 1 / \log\log(t)$.

### Prediction (4) — Gray-code factor

- Standard binary encoding: $\tau_{\text{int}}^{\text{binary}} \sim \exp(\beta E_{\text{cliff}})$
  where $E_{\text{cliff}}$ is the artificial cliff barrier from
  multi-spin-flip transitions across binary cell boundaries.
- Gray code: $\tau_{\text{int}}^{\text{gray}} \sim \exp(\beta E_{\text{real}})$
  where $E_{\text{real}}$ is the genuine thermodynamic barrier of the
  underlying continuous energy.
- **Improvement factor:** $\tau_{\text{int}}^{\text{binary}} / \tau_{\text{int}}^{\text{gray}} \approx \exp(\beta (E_{\text{cliff}} - E_{\text{real}}))$.
  Exponential improvement.
- The leading coefficient depends on the binary cell width $\Delta$
  and the local Hamiltonian's coupling strength.

## Where we suspect Deep Think will surprise us

- **Tighter constants in (3).** Our $1/\log(t)$ guess might be too
  conservative or too aggressive. Deep Think might find a polynomial
  schedule $\delta_t \sim t^{-1/2}$ via more careful coupling.
- **Necessary condition for (1).** Our suspected necessary-and-
  sufficient condition is plausible but unverified. Deep Think
  might find a weaker sufficient condition (good news) or a more
  restrictive necessary condition (bad news for deployment).
- **Replica-spacing in (2).** Maybe the optimal isn't 0.23 in this
  context — that's the Roberts-Rosenthal limit for IID-target
  Metropolis. Verifier-filtered sampling might have a different
  optimum.
- **Gray-code factor in (4).** The exponential-improvement claim
  might be too optimistic. Could be polynomial in some regimes,
  or might break down at extremely low T.

## Post-response action plan

After Deep Think replies:

1. Diff the response against pre-registered predictions above.
2. Update `zenil-alpha-verifier-derivation.md` with confirmed/refuted
   bounds.
3. If (1) gives a measurable test, ship a `python/carnot/eval/phi_test.py`
   module that estimates $\Phi$ on held-out data.
4. If (3) gives leading constants, update `_bmad/architecture.md`
   throughput-vs-depth table with concrete bounds.
5. If (4) confirms exponential Gray-code improvement, that's
   publishable as a Phase 2 transpiler theorem (REQ-PHASE2-006).

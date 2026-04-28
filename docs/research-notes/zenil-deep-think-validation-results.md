# Deep Think Validation Results — Corrected Math

**Status:** Authoritative. Supersedes the Gemini-Round-6 results in
`zenil-deep-think-round6-prompt.md`.
**Date:** 2026-04-28 (evening)
**Prompt:** `zenil-deep-think-validation-prompt.md`

Deep Think's independent re-derivation refuted four of five Gemini
claims (Q1, Q3, Q4, Q5) and corrected an algebra error in Q2. Most
significantly, Deep Think identified a **fatal weakest link** that
no prior derivation had surfaced: the *orthogonality stall*.

## Q1 — Structural Φ — REFUTED

**Gemini claim:** Φ = spectral gap of Langevin generator $\mathcal{L}_E$;
sufficient condition $\int B \, \nabla \log \mu_P \, d\mu_P < 0$.

**Why it fails:** Category error. The verifier is a *discrete pointwise
reweighting* (importance sampling / rejection), not a continuous Langevin
diffusion. The integral inequality is also dimensionally invalid — it
integrates a scalar against a gradient vector, yielding a vector that
cannot be "$<0$" as a strict inequality.

**Correct derivation (Deep Think):** From the first-order $L^2(\mu_P)$
expansion of the squared-norm contraction with high-T Taylor of the
filter $w(x) \approx 1 - (E - \mathbb{E}_{Q_t}[E])/T$, equating to the
target ansatz $\alpha_t^{\text{eff}} = \Phi/(T\delta_t)$ isolates:

$$
\boxed{\Phi = \frac{1}{\delta_t}\, \mathrm{Cov}_{Q_t}\!\left(\frac{Q_t}{\mu_P},\, E\right)}
$$

**Measurable sufficient condition for Φ > 0:** the density-overestimation
ratio $Q_t/\mu_P$ must have strictly positive covariance with the
verifier energy $E$ under the proposal $Q_t$. Physically: *the verifier
must systematically assign higher energy penalties to the modes where
the model is currently hallucinating*.

**Deployment:** estimable from samples — generate $n$ samples from $Q_t$,
estimate $\widehat{Q_t/\mu_P}$ pointwise (importance ratio if $\mu_P$
density available, or KDE / one-sample test otherwise), compute
$\widehat\Phi = \frac{1}{\delta_t}\widehat{\mathrm{Cov}}_{Q_t}(Q_t/\mu_P, E)$
with bootstrap CI.

## Q2 — Stochastic MCMC bound — REFUTED algebra, REFINED scaling

**Gemini claim:** $N_t > \sigma^2 \tau_{\text{int}} / (2\alpha_t \Phi \delta_t^2)$,
which when $\alpha_t = \Phi/(T\delta_t)$ becomes $\sigma^2 T\tau / (2\Phi^2\delta_t)$.

**Why it fails:** Spurious extra $\Phi$ in the base formula's denominator,
which cascaded into the $\Phi^2$ in the substituted form.

**Correct derivation (Deep Think):** From $(1 - 2\alpha_t^{\text{eff}})\delta_t^2 + \sigma^2 \tau_{\text{int}}/N_t < \delta_t^2$:

$$
N_t > \frac{\sigma^2 \tau_{\text{int}}}{2 \alpha_t^{\text{eff}} \delta_t^2}
$$

Substituting $\alpha_t^{\text{eff}} = \Phi/(T\delta_t)$:

$$
\boxed{N_t > \frac{\sigma^2 \, T \, \tau_{\text{int}}}{2 \, \Phi \, \delta_t}}
$$

**Linear** in $1/\delta_t$ in the deployed parameterisation. The $1/2$
constant is correct (cross-term of $(1-\alpha)^2 = 1 - 2\alpha + \alpha^2$).

**Phase-transition rescue:** at critical points where $\tau_{\text{int}} \to \infty$
(critical slowing down), local MCMC strictly halts. The rescue is
**Parallel Tempering / Replica Exchange** — bridges the slow-mixing
low-T chains via swap proposals from high-T chains.

## Q3 — Annealing schedule — REFUTED

**Gemini claim:** $T_t = T_0 \log t_0 / \log t$, $N_t = N_0 (\log t / \log t_0)^2$,
logarithmic convergence.

**Why it fails:** Logarithmic cooling is the schedule for *single-chain
simulated annealing* targeting global minimisation. It actively breaks
the verifier-filter framework:

1. As $T \to 0$, the high-T Taylor expansion underlying Q1/Q2 becomes
   invalid.
2. As $T \to 0$, the third contractivity prerequisite ($T > T_{\text{crit}}$)
   is violated. Below $T_{\text{crit}}$, the filtered distribution suffers
   Dirac mode-collapse onto the energy minimum.
3. Q2's rescue invokes Parallel Tempering, which already eliminates
   mixing bottlenecks at constant temperature — making aggressive
   cooling unnecessary.

**Correct schedule (Deep Think):** hold $T$ constant at $T_{\text{floor}} \gtrsim T_{\text{crit}}$.
Linear finite-time convergence:

$$
T_t = T_{\text{floor}}, \qquad \boxed{N_t = \frac{\sigma^2 \, T_{\text{floor}} \, \tau_{\text{int}}}{2 \, \Phi \, \delta_t}}
$$

Per-step contraction is constant: $\Delta\delta_t \approx -\Phi/T_{\text{floor}}$.
Convergence is *linear in $t$*, terminating at the structural noise floor
$\mathcal{O}(\varepsilon)$ in finite time.

**This is much better news than the logarithmic claim.** Constant-T
training with PT bridging is the deployed schedule.

## Q4 — Gray-code factor — REFUTED

**Gemini claim:** $\Gamma = \exp((\kappa/T)\sum_{i=1}^{k}(2^i - 1))$.
For k=8: factor 1004.

**Why it fails:**

- **Physics:** by Kramers' escape rate theory, MCMC transition time is
  dictated by the *single largest* energy barrier ($\max \Delta E$),
  not the sum of all intermediate barriers. Summing barrier exponents
  multiplies sequential-bottleneck probabilities, violating Markov
  properties.
- **Arithmetic:** $\sum_{i=1}^{8}(2^i - 1) = 502$, not 1004. Gemini
  double-counted.

**Correct derivation (Deep Think):** standard binary's worst transition
crosses the halfway threshold (e.g., $127 \to 128$), which flips the
MSB. For Hamiltonian coupling $\kappa$, this single flip induces an
artificial saddle-point jump of $\Delta E_{\max} \approx \kappa \cdot 2^{k-1}$.
Gray code bounds the single-step jump to $\kappa$. The improvement
factor is:

$$
\boxed{\Gamma \approx \exp\!\left(\frac{\kappa(2^{k-1} - 1)}{T}\right)}
$$

For $k=8$: exponent ≈ 127 (not 502, not 1004). Still exponential — and
still the basis of a publishable Phase 2 transpiler theorem — but with
the correct constant.

**Low-T plateau:** at extreme low $T$, genuine topological macro-barriers
of the underlying Hamiltonian overtake the artificial-encoding
bottlenecks; the speedup naturally saturates.

## Q5 — PT acceptance rate — REFUTED

**Gemini claim:** target 0.35 acceptance for verifier-filtered sampling.

**Why it fails:** confused intra-chain spatial proposals (where MALA's
optimal is $\sim 0.57$) with inter-chain thermodynamic swaps (where
PT's optimal is $\sim 0.234$).

PT swap acceptance evaluates *existing* states against thermodynamic
overlap — no new spatial sampling involved. By CLT, macroscopic energies
in high dimensions are Gaussian, so the optimal swap acceptance between
adjacent Gaussians for maximum temperature diffusion is rigorously
**0.234** (Roberts-Rosenthal-Gilks).

**Action:** keep PT swap acceptance at 0.234. Do NOT ship 0.35 as a
constant. The verifier-filter biases proposals (which IS where higher
acceptance might apply, e.g., for MALA-style verifier-filtered samplers
within each chain), but PT *between* chains stays at 0.234.

## Final integration check — NOT consistent

Gemini's five claims form a web of contradictions:

- **Dimensional:** Q1 defines Φ with units of (1/time) (spectral gap),
  Q2 treats it as dimensionless. Cannot both be right.
- **Algorithmic:** Q3 assumes log cooling (single-chain SA), Q5 assumes
  PT (multi-chain) — incompatible algorithms.
- **Mathematical:** Q3's $T \to 0$ explicitly violates Q1/Q2's $T > T_{\text{crit}}$
  high-T prerequisite.

The corrected math (above) is internally consistent: Φ is dimensionless,
constant-T schedule preserves Taylor validity, PT bridges
$\tau_{\text{int}}$ blowup, swap acceptance stays at 0.234.

## The fatal weakest link — orthogonality stall (NEW)

**The most consequential finding.** Even with all corrected math,
Deep Think identified a fundamental limit:

The corrected $\Phi$ from Q1 is $\Phi \propto \mathrm{Cov}_{\mu_P}(g, E)$
where $g = (Q_t - \mu_P)/(\delta_t \mu_P)$ is the normalised error
profile. The verifier provides a restorative force *only along
eigenvectors where $g$ is correlated with $E$*.

As self-distillation runs, the model rapidly eliminates the "easy"
errors — those aligned with $E$. The residual error vector $g$
**inevitably rotates into the null-space of $E$** (becomes orthogonal
to the verifier's constraint landscape).

When orthogonality occurs:

$$
\mathrm{Cov}_{\mu_P}(g, E) = 0 \implies \Phi = 0
$$

— even when $\delta_t$ is still large. Once $\Phi$ falls below the
verifier's noise bound $\varepsilon$, the restorative step collapses.
**The self-distillation loop permanently stalls at an irreducible
orthogonality plateau, and infinite scaling of $N_t$ cannot rescue it.**

### Implications for Carnot

This is a fundamental architectural finding:

1. **Single-verifier self-distillation has a hard ceiling** that is
   *not* a compute problem. Adding samples / hardware acceleration
   cannot push past it.
2. **The Phase 2 hardware mandate** (`_bmad/architecture.md`) is still
   mathematically justified for the *pre-orthogonality* regime
   ($\delta_t \to \delta_{\text{plateau}}$ where plateau depends on
   $E$'s coverage of the residual error space). It does not enable
   $\delta_t \to 0$ as previously claimed.
3. **The Phase 3 foundation-model architecture must include
   adversarial verifier rotation** to escape the orthogonality stall:
   - Periodically swap the constraint set $E$ with a different one
     covering different residual error modes.
   - Use *multiple* verifiers spanning the constraint space; combine
     their gradients so $g$ cannot be orthogonal to all of them
     simultaneously.
   - Detect the orthogonality stall in-loop (Φ → 0 estimator) and
     switch verifiers as a regime-change signal.

### Mitigations to ship

- **Adversarial rotation of $E$**: at training time, rotate among
  several constraint sets ($E_1, E_2, ..., E_k$). The covariance
  $\mathrm{Cov}_{\mu_P}(g, E_j)$ is nonzero for at least one $j$ as long
  as the union of the $E_j$'s spans the residual-error space.
- **Multi-verifier ensemble**: stack verifiers; compute Φ for each;
  use the verifier with maximum Φ at each round.
- **Phi-stall detector**: instrument the self-distillation loop to
  estimate Φ via the corrected formula above and halt training (or
  swap verifier) when $\Phi$ falls below a threshold.

## What this changes in the Carnot stack

Updates required:

1. **`zenil-grounded-self-distillation-deployable-stack.md`** (the change
   proposal): rewrite Exp A (phi_test.py) with the *correct* formula
   $\widehat\Phi = \frac{1}{\delta_t}\widehat{\mathrm{Cov}}_{Q_t}(Q_t/\mu_P, E)$.
   Rewrite Exp B (anneal.py) to use *constant-T floor* schedule, not
   logarithmic. Rewrite Exp C to keep PT acceptance at 0.234 and
   remove the 0.35 hyperparameter. Rewrite Exp D's REQ-PHASE2-006
   with the correct factor $\exp(\kappa(2^{k-1}-1)/T)$.

2. **`_bmad/architecture.md` "Asymptotic Hardware Mandate"**: update
   the throughput-vs-depth table — the bound is $N \sim T/\delta_t$
   (linear), and the Gray-code factor is exp(κ·2^(k-1)/T) (not the
   super-exponential super-claim). The hardware mandate stands but
   the constants need recomputing.

3. **NEW change proposal needed:** orthogonality-stall mitigation.
   Adversarial rotation of $E$, multi-verifier ensembles, phi-stall
   detection. This is Phase 3 architecture work, not Phase 2.

4. **NEW memory entry:** the orthogonality stall is the single most
   important strategic finding from this work. Should be a project
   memory cross-linking to the Phase 3 architecture decisions.

## Take-aways

- **Listen to Deep Think over Gemini for math.** Gemini was confidently
  wrong on 4 of 5 claims and arithmetically wrong on Q4 (1004 vs 502 —
  basic addition error). Iterative critique with a less capable model
  is a *prompt-engineering* tool, not a mathematical-derivation tool.
- **Cross-validation discipline works.** We pre-registered predictions
  before sending to Deep Think; the diff exposed how confidently-wrong
  the prior reasoning was.
- **The orthogonality stall is publishable.** "Self-distillation under
  fixed-verifier filter has a fundamental ceiling determined by the
  rotation of residual error into the verifier's null space"
  is — to my knowledge — not in the literature in this exact form.
  Worth a paper. Mitigations (adversarial verifier rotation) are also
  publishable.
- **Phase 2 hardware mandate survives** but is reframed: it enables
  reaching the orthogonality plateau efficiently, not driving
  $\delta_t \to 0$ in absolute terms. The argument for sovereign
  high-throughput sampling stands.
- **Phase 3 foundation model needs adversarial verifier rotation** as
  a load-bearing architectural feature.

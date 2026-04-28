# Zenil-Grounded Self-Distillation: Deployable Stack

**Status:** Draft change proposal. Requested for milestone 2026.04.81 or .82.
**Origin:** 2026-04-28 Round-6 Deep Think response on the Zenil α_t ×
Carnot verifier formalism. Three rounds of Gemini ↔ Carnot critique
produced a deployable contractivity bound for verifier-filtered
self-distillation. Deep Think confirmed Carnot's pre-registered
predictions and upgraded several to more powerful forms (spectral-gap
Φ, super-exponential Gray-code factor, 0.35 PT acceptance rate, log-
convergence schedule). Full math in
`docs/research-notes/zenil-alpha-verifier-derivation.md` and
`docs/research-notes/zenil-deep-think-round6-prompt.md`.

**Target milestone:** **2026.04.81** preferred; .82 acceptable if .81
is over-budget on infrastructure work.

**Priority:** **High.** The Phase 2 hardware mandate
(`_bmad/architecture.md` → "Asymptotic Hardware Mandate") is now
mathematically justified. This proposal ships the four code artifacts
that the math demands. Without them, every FR-11 self-distillation
experiment risks silent collapse as $\delta_t \to 0$; with them,
contractivity becomes measurable, deployable, and — for the Gray-code
factor — publishably falsifiable.

**Depends on:** nothing. All math derivations are landed
(commits `f25c697e`, `2641fc44`, `8b2fd55a`, `eb643f7e`, `1640aee1`).
This proposal is pure code-side follow-up.

## Summary

Four targeted code artifacts that operationalise the Round-6 result:

1. **`python/carnot/eval/phi_test.py`** — Φ > 0 measurement on held-out
   evaluation data. The structural restorative force is the load-bearing
   precondition for verifier-filtered contractivity; without an in-loop
   Φ test, FR-11 experiments cannot detect when the verifier signal has
   degenerated.
2. **`python/carnot/training/anneal.py`** — deployable joint
   $T_t / N_t$ annealing schedule. Round-6 gave the exact formula:
   $T_t = T_0 \log t_0 / \log t$, $N_t = N_0 (\log t / \log t_0)^2$.
   Used by FR-11 self-distillation loops and Phase 2 hardware
   orchestration.
3. **`PT_TARGET_ACCEPTANCE = 0.35`** in
   `python/carnot/hardware/transpiler/distill.py` — directly shippable
   hyperparameter for verifier-filtered parallel tempering. Replaces
   the implicit Roberts-Rosenthal 0.23 default; cite Round-6 in code
   comment.
4. **REQ-PHASE2-006 Gray-code theorem experiment.** Falsifiable
   measurement of $\tau_{\text{int}}^{\text{binary}} / \tau_{\text{int}}^{\text{gray}}$
   on identical Hamiltonian, two encodings. If the super-exponential
   improvement factor $\Gamma = \exp((\kappa/T)\sum_{i=1}^{k}(2^i-1))$
   holds empirically, that's a publishable Phase 2 result.

## What this proposal IS NOT

- **Not a refactor of the existing transpiler.** All three modules
  (`gray_code.py`, `distill.py`, `diagnostics.py`) shipped 2026-04-27.
  This proposal adds *new* artifacts and one constant.
- **Not a foundation-model self-distillation deployment.** The math
  describes Phase 3 dynamics; the artifacts ship Phase 1 measurement
  + Phase 2 hyperparameter. Deployment of the actual self-distillation
  loop is gated on later milestones.
- **Not a publication.** REQ-PHASE2-006 is *the experiment* for a
  publication; the actual writeup is downstream.

## Proposed experiments

### Exp A — Φ > 0 measurement module

**Deliverable:** `python/carnot/eval/phi_test.py` +
`tests/python/test_phi_test.py` +
`results/experiment_<N>_phi_test_module.json`.

**What it does:**

1. Implement `phi_test(Q_samples, mu_P_samples, log_mu_P) -> PhiTestResult`
   returning:
   - Estimated Φ via the bias-score inner product:
     $\widehat\Phi = -\frac{1}{2}\widehat{\mathbb{E}}[\mathrm{sgn}(B(x)) \cdot (\log \mu_P(x) - \overline{\log \mu_P})]$
     where $B(x) = \log Q_t(x) - \log \mu_P(x)$.
   - Bootstrap CI (1000 resamples) for $\widehat\Phi$.
   - Hypothesis test: reject $H_0: \Phi \le 0$ at $\alpha = 0.05$.
2. Optional: spectral-gap proxy via Langevin generator power iteration
   when $\nabla \log \mu_P$ is available analytically (used for Carnot's
   constraint-derived energies).
3. 15+ unit tests covering: monotone-likelihood case (Φ > 0), uniform
   error case (Φ ≈ 0), miscalibrated-mode case (Φ < 0).

**Acceptance:** module imports from `carnot.eval.phi_test`, tests pass,
artifact reports `phi_test_module_complete`.

### Exp B — Joint annealing schedule module

**Deliverable:** `python/carnot/training/anneal.py` +
`tests/python/test_anneal_schedule.py` +
`results/experiment_<N>_annealing_module.json`.

**What it does:**

1. Implement `zenil_schedule(t, t_0, T_0, N_0) -> AnnealStep` returning
   `(T_t, N_t)` per the Round-6 formula:
   $$T_t = T_0 \frac{\log t_0}{\log t}, \quad N_t = N_0 \left(\frac{\log t}{\log t_0}\right)^2$$
2. Coupled budget mode: when current $\delta_t$ estimate is provided,
   return $N_t = \lceil \sigma^2 T_t \tau_{\text{int}} / (2 \Phi^2 \delta_t)\rceil$
   directly (the deterministic-α coupled bound).
3. Hardware-tier mode: given a target tier (CPU / KV260 / Extropic /
   photonic) with samples-per-second cap, returns the minimum $\delta_t$
   the tier can sustain at a given $T_t$.
4. 12+ unit tests covering monotone $T_t$ decrease, monotone $N_t$
   increase, hardware-tier saturation thresholds.

**Acceptance:** module callable from FR-11 experiments, tests pass,
hardware-tier table matches `_bmad/architecture.md`.

### Exp C — PT acceptance rate hyperparameter

**Deliverable:** edit
`python/carnot/hardware/transpiler/distill.py` to set
`PT_TARGET_ACCEPTANCE = 0.35` with code-comment citation +
`tests/python/test_transpiler_distill.py` updated assertion +
`results/experiment_<N>_pt_acceptance_v035.json`.

**What it does:**

1. Replace any implicit 0.23 default with the explicit constant
   `PT_TARGET_ACCEPTANCE = 0.35`.
2. Comment cites Round-6 Deep Think result: verifier-filtered
   sampling shifts the acceptance-reward trade-off toward higher
   temperatures; 0.35 prioritises mode-hopping over local convergence.
3. Adaptive temperature-spacing logic targets the new acceptance rate.
4. Existing test `test_distiller_train_epoch_returns_metrics` updated
   to verify the new constant; new test asserts that PT runs achieve
   acceptance ≈ 0.35 ± 0.05 on a synthetic bimodal target.

**Acceptance:** transpiler PT runs hit 0.35 ± 0.05 acceptance on the
synthetic target; existing tests pass.

### Exp D — REQ-PHASE2-006 Gray-code factor measurement

**Deliverable:** `python/carnot/hardware/transpiler/measurements.py`
(new) + `scripts/experiment_<N>_gray_code_factor.py` +
`tests/python/test_gray_code_factor.py` +
`results/experiment_<N>_gray_code_tau_int_factor.json` +
new spec entry REQ-PHASE2-006 in
`openspec/capabilities/hardware-transpiler/spec.md`.

**What it does:**

1. Run identical Hamiltonian (synthetic 2D Gaussian mixture or
   continuous Ising) under standard binary encoding vs Gray-code
   encoding, both at 8-bit width, identical PT-PCD setup.
2. Measure $\tau_{\text{int}}$ for each via integrated-autocorrelation
   estimation (already in `diagnostics.py`).
3. Compute empirical $\widehat\Gamma = \widehat\tau_{\text{int}}^{\text{binary}} / \widehat\tau_{\text{int}}^{\text{gray}}$.
4. Compare to theoretical $\Gamma = \exp((\kappa/T)\sum_{i=1}^{8}(2^i - 1))$
   for the lattice's measured coupling $\kappa$ and temperature $T$.
5. Report empirical vs theoretical ratio; pass condition is empirical
   $\ge$ theoretical / 10 (allowing for sub-leading corrections).
6. New spec REQ-PHASE2-006: "The continuous-to-Ising transpiler's
   Gray-code visible-spin encoder reduces integrated autocorrelation
   time by factor $\Gamma = \exp((\kappa/T)\sum_{i=1}^{k}(2^i - 1))$
   vs standard binary encoding at bit width $k$."

**Acceptance:** experiment artifact reports
`gray_code_factor_confirmed` if empirical $\widehat\Gamma$ exceeds the
threshold; else `gray_code_factor_below_theoretical` with diagnostic
log of which terms in the Hamiltonian dominate.
This is a falsifiable Phase 2 transpiler theorem result.

## Decentralization implications

- **Rule 1 (local-first):** unaffected. All four artifacts run on
  CPU/ROCm; closed-weight LLMs are not in the verifier path.
- **Rule 5 (hardware portability as political requirement):**
  *strengthened*. The math justifies the FPGA / Extropic / photonic
  tracks not as performance optimisations but as *necessary
  infrastructure for sovereignty in self-distillation*. REQ-PHASE2-006
  is the empirical anchor for that claim.
- **Rule 7 (no vendor abstractions in core):** unaffected. All
  artifacts are pure-numpy / pure-JAX.

## Risks

- **Φ test estimation variance.** Bootstrap CI on small held-out sets
  may be too wide to reject $H_0$ at $\alpha=0.05$. Mitigation: the
  test reports the CI; experiments using it can choose their own
  significance level.
- **Annealing-schedule constants.** Round-6 gave the formula but the
  *leading* $T_0, N_0$ are problem-specific. Experiments using
  `zenil_schedule` must calibrate per task; the module exposes a
  `calibrate()` helper that uses early-iteration $\delta$ measurements
  to pin $T_0, N_0$.
- **PT acceptance 0.35 may not generalise.** Round-6's number applies
  to verifier-filtered sampling; non-filtered or differently-filtered
  PT may want different targets. Mitigation: the constant is named
  clearly and the test asserts against a synthetic verifier-filtered
  target.
- **REQ-PHASE2-006 may falsify.** The super-exponential factor is
  bold; sub-leading corrections from the underlying Hamiltonian
  may dominate. Mitigation: the acceptance threshold is generous
  (empirical $\ge$ theoretical / 10) and a falsifying outcome is
  still publishable as a "Gray code helps but not as much as
  predicted" result.

## Acceptance criteria

1. `python/carnot/eval/phi_test.py` exists, 15+ tests pass, returns
   bootstrap CI for $\Phi$.
2. `python/carnot/training/anneal.py` exists, 12+ tests pass, three
   modes (formula / coupled-budget / hardware-tier) all callable.
3. `PT_TARGET_ACCEPTANCE = 0.35` constant lives in `distill.py` with
   Round-6 citation in comment; PT runs hit 0.35 ± 0.05 acceptance
   on synthetic bimodal target.
4. Empirical Gray-code $\widehat\Gamma$ measured on identical
   Hamiltonian, two encodings, 8-bit; either confirms the theoretical
   factor (within /10) or produces a publishable falsification.
5. New spec entry REQ-PHASE2-006 added to
   `openspec/capabilities/hardware-transpiler/spec.md`.
6. Cross-references to `docs/research-notes/zenil-*.md` in all four
   modules so future readers can trace the math.

## Why this is in change-proposals, not a single experiment

The four artifacts cluster around a single mathematical result but
target different deployment surfaces (eval, training, transpiler,
spec). Bundling them as one experiment would mix scope; splitting
into one-experiment-per-artifact would lose the unifying narrative
of "deployable Round-6". The proposal pattern preserves both: each
exp ships independently but they share a common origin in the
Zenil/Round-6 derivation.

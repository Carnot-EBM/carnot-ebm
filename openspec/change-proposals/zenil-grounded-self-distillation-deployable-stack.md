# Zenil-Grounded Self-Distillation: Deployable Stack

**Status:** Draft change proposal. **MAJOR REVISION 2026-04-28 evening**
after Deep Think validation refuted four of five Gemini-derived claims.
Requested for milestone 2026.04.81 or .82 with the corrected math.

**REVISION SUMMARY:**
- Q1 Φ formula corrected: spectral-gap reframing was wrong (verifier
  is discrete pointwise reweighting, not Langevin diffusion). Correct:
  $\Phi = \frac{1}{\delta_t} \mathrm{Cov}_{Q_t}(Q_t/\mu_P, E)$.
- Q2 algebra fixed: $N_t > \sigma^2 T \tau_{\text{int}} / (2\Phi \delta_t)$
  (linear, single Φ).
- Q3 schedule replaced: constant-T floor, NOT logarithmic cooling.
  Linear convergence $\Delta\delta_t = -\Phi/T_{\text{floor}}$.
- Q4 factor corrected: $\Gamma = \exp(\kappa(2^{k-1}-1)/T)$, exponent ≈
  127 at k=8 (Gemini's 1004 was a double-counted arithmetic error AND
  a Kramers-physics violation).
- Q5 PT acceptance unchanged from standard 0.234. The 0.35 Gemini number
  was a hallucination conflating MALA and PT. **Drop the Exp C
  hyperparameter change.**

**NEW LOAD-BEARING FINDING (Deep Think):** the *orthogonality stall* —
single-verifier distillation has a fundamental compute-immune ceiling
when residual error rotates into the verifier's null space. Adds a
NEW Exp E (multi-verifier rotation) to this proposal. See
`docs/research-notes/zenil-deep-think-validation-results.md` and
memory entry `project_orthogonality_stall.md`.
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

### Exp A — Φ > 0 measurement module (CORRECTED 2026-04-28)

**Deliverable:** `python/carnot/eval/phi_test.py` +
`tests/python/test_phi_test.py` +
`results/experiment_<N>_phi_test_module.json`.

**What it does:**

1. Implement `phi_test(Q_samples, log_mu_P_fn, E_fn, delta_t) -> PhiTestResult`
   returning:
   - Estimated Φ via the corrected covariance formula:
     $\widehat\Phi = \frac{1}{\delta_t} \widehat{\mathrm{Cov}}_{Q_t}\!\left(\frac{Q_t}{\mu_P},\, E\right)$
     where the importance ratio $Q_t/\mu_P$ is estimated pointwise via
     `log_mu_P_fn(x) - log_Q_t(x)` and $E$ via `E_fn(x)`.
   - Bootstrap CI (1000 resamples) for $\widehat\Phi$.
   - Hypothesis test: reject $H_0: \Phi \le 0$ at $\alpha = 0.05$.
2. Detection of the *orthogonality stall*: when $\widehat\Phi$ falls
   below a configured noise threshold $\varepsilon$, return
   `PhiTestResult(stalled=True, reason='orthogonality')` so the
   self-distillation loop can halt or rotate verifier.
3. 15+ unit tests covering:
   - Aligned-error case (Φ > 0): hallucination modes correlate with E.
   - Orthogonal-error case (Φ ≈ 0): residual rotated into E's null space.
   - Anti-aligned case (Φ < 0): verifier reinforces hallucinations.

**Acceptance:** module imports from `carnot.eval.phi_test`, tests pass,
artifact reports `phi_test_module_complete`.

**DROPPED FROM ORIGINAL DRAFT:** the spectral-gap proxy. The
spectral-gap reframing was a Gemini category error (Deep Think Q1
verdict). Phi is a covariance, not an eigenvalue.

### Exp B — Constant-T schedule module (CORRECTED 2026-04-28)

**Deliverable:** `python/carnot/training/anneal.py` +
`tests/python/test_anneal_schedule.py` +
`results/experiment_<N>_annealing_module.json`.

**What it does:**

1. Implement `constant_t_schedule(t, T_floor, sigma2, tau_int, Phi, delta_t) -> AnnealStep`
   returning `(T_t, N_t)` per the corrected formula:
   $$T_t = T_{\text{floor}}, \qquad N_t = \left\lceil \frac{\sigma^2 \, T_{\text{floor}} \, \tau_{\text{int}}}{2 \, \Phi \, \delta_t}\right\rceil$$
   with $T_{\text{floor}} \gtrsim T_{\text{crit}}$ provided by the caller.
2. `T_floor` calibration helper: estimates $T_{\text{crit}}$ from the
   energy landscape (heuristic: roughly the temperature at which the
   filter's KL divergence to $\mu_P$ is half its high-T limit).
3. Hardware-tier mode: given a target tier with samples-per-second
   cap, returns the minimum $\delta_t$ the tier can sustain *before
   reaching the orthogonality plateau* (delta_plateau).
4. 12+ unit tests covering: $T_t$ stays constant; $N_t$ scales as
   $1/\delta_t$ (linear); hardware tiers correctly compute
   plateau-attainable depth.

**Acceptance:** module callable from FR-11 experiments, tests pass,
hardware-tier table matches `_bmad/architecture.md`.

**DROPPED FROM ORIGINAL DRAFT:** logarithmic cooling. Deep Think Q3
verdict: log cooling actively breaks the high-T Taylor expansion
underlying Q1/Q2 and is unnecessary because PT bridges
$\tau_{\text{int}}$ blowup at constant T.

### Exp C — DROPPED (2026-04-28)

**Original scope:** ship `PT_TARGET_ACCEPTANCE = 0.35`.

**Why dropped:** Deep Think Q5 verdict refuted the 0.35 number as a
Gemini hallucination conflating intra-chain MALA acceptance ($\sim 0.57$)
with inter-chain PT swap acceptance (Roberts-Rosenthal-Gilks $\sim 0.234$).
PT swaps evaluate existing states by thermodynamic overlap; CLT-Gaussian
energies in high dimensions give the rigorous 0.234 optimum
*regardless of how the proposals were generated*. The verifier-filter
biases proposals (where a higher acceptance like 0.57 might apply for
within-chain MALA), but PT *between* chains stays at 0.234.

**Action:** keep PT swap acceptance at 0.234. No code change required.
The current transpiler (`python/carnot/hardware/transpiler/distill.py`)
already defaults to 0.234 implicitly via Roberts-Rosenthal-Gilks; ensure
the constant `PT_TARGET_ACCEPTANCE = 0.234` is *explicitly* named with
a code-comment citation so future readers don't re-introduce the
0.35 confusion.

### Exp D — REQ-PHASE2-006 Gray-code factor measurement (CORRECTED 2026-04-28)

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
4. Compare to **corrected** theoretical $\Gamma = \exp(\kappa(2^{k-1} - 1)/T)$
   per Kramers' escape rate (single-largest-barrier, not summed). For
   $k = 8$, theoretical exponent = 127.
5. Report empirical vs theoretical ratio; pass condition is empirical
   $\ge$ theoretical / 10 (allowing for sub-leading corrections).
6. New spec REQ-PHASE2-006: "The continuous-to-Ising transpiler's
   Gray-code visible-spin encoder reduces integrated autocorrelation
   time by factor $\Gamma \approx \exp(\kappa(2^{k-1} - 1)/T)$
   vs standard binary encoding at bit width $k$, where $\kappa$ is the
   underlying Hamiltonian's coupling strength."

**Acceptance:** experiment artifact reports
`gray_code_factor_confirmed` if empirical $\widehat\Gamma$ exceeds the
threshold; else `gray_code_factor_below_theoretical` with diagnostic
log of which terms in the Hamiltonian dominate.
This is a falsifiable Phase 2 transpiler theorem result.

**CORRECTED 2026-04-28:** Gemini's super-exponential factor
$\exp((\kappa/T)\sum(2^i - 1))$ was wrong on both physics (Kramers
depends on max barrier, not sum) and arithmetic (Σ for k=8 = 502,
not 1004). Correct factor is exponential in $2^{k-1}$, still publishable
as a Phase 2 theorem.

### Exp E — Multi-verifier pre-emptive rotation (NEW 2026-04-28; UPDATED Round-7)

**Round-7 update:** Deep Think confirmed that **rotation strictly
dominates ensemble combine** asymptotically — ensemble combine creates
multi-modal glassy frustrated MCMC landscapes ($\tau_{\text{int}}^{\text{ens}}$
blows up factorially). Pre-emptive rotation (every round, $\arg\max_i \widehat\Phi_i$)
is strictly optimal — staying causes exponential Φ decay. Rejected
all ensemble-combine architecture from this proposal.

**Round-7 also identified the load-bearing security finding:** the Q3
discriminating test has ZERO POWER against null-space mimicry attacks.
Adversarial specification gaming on Boolean $E$ (e.g., Z3 satisfaction)
produces $\mathrm{Cov}_{\mu_P}(g_t, E^m) = 0$ for all $m$, perfectly
mimicking the orthogonality stall. Multi-verifier pre-emptive rotation
is the **unique** mathematically-rigorous defence.

### Exp E — orig draft retained below (now superseded by E.rev)

**Deliverable:** `python/carnot/training/multi_verifier.py` (new) +
`tests/python/test_multi_verifier_rotation.py` +
`results/experiment_<N>_orthogonality_stall_validation.json` +
`results/experiment_<N>_multi_verifier_rotation_breaks_stall.json`.

**Why this experiment exists:** Deep Think identified the *orthogonality
stall* as a fundamental compute-immune ceiling for single-verifier
self-distillation. As the model eliminates errors aligned with $E$,
the residual rotates into $E$'s null space and $\Phi \to 0$. Infinite
$N_t$ cannot rescue this. Mitigation: adversarial rotation among
multiple verifiers $E_1, ..., E_k$ whose union spans the residual-error
space.

**What it does:**

1. **Stall validation.** Run a single-verifier self-distillation loop
   on a synthetic target with $E$ covering only a subspace of the
   residual error space. Measure $\widehat\Phi$ each round via Exp A's
   `phi_test`. Verify the orthogonality stall: $\widehat\Phi$ decays
   to zero while $\delta_t$ remains bounded above $\delta_{\text{plateau}}$.
   Artifact reports `stall_validated` with $(\delta_{\text{plateau}}, t_{\text{stall}})$.

2. **Multi-verifier rotation.** Implement
   `MultiVerifierRotator(verifiers: list[E_i])` that:
   - At each round, computes $\widehat\Phi_i$ for each verifier $E_i$.
   - Selects the verifier with maximum $\widehat\Phi_i$ for the next
     round's filter.
   - Rotates verifiers automatically when current $\widehat\Phi_i$
     falls below $\varepsilon$.

3. **Stall-break demonstration.** Run the same synthetic distillation
   loop with the rotator. Demonstrate that $\delta_t$ continues to
   decrease past $\delta_{\text{plateau}}$ from the single-verifier
   case. Artifact reports `multi_verifier_breaks_stall` with the
   delta improvement.

4. **Multi-verifier ensemble combine.** Optional: when several
   $\widehat\Phi_i$'s are simultaneously positive, combine the filter
   gradients additively rather than rotating. Test that the combined
   filter improves over any single verifier on synthetic data.

5. 12+ unit tests covering: rotation triggers on stall detection,
   max-Φ selection logic, ensemble-combine math, regression test
   for stall-break.

**Acceptance:**
- Stall validation: empirical $\delta_{\text{plateau}}$ matches
  theoretical orthogonality threshold.
- Stall-break: $\delta_t$ continues decreasing under rotation, with
  improvement factor at least 2× over the single-verifier plateau.
- Tests pass.

**Spec impact:** new REQ-PHASE3-001: "Phase 3 self-distillation must
include multi-verifier rotation to escape the orthogonality stall.
Single-verifier filtering converges only to $\delta_{\text{plateau}} > 0$
determined by the residual error's projection onto $E$'s null space."

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

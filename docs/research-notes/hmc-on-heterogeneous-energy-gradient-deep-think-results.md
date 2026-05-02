# Deep Think Response — HMC Compatibility on Carnot's Heterogeneous-Lipschitz ∇E

**Status:** Response received 2026-05-02 ~07:50Z. Methodology-only with
explicit self-correction at the end. 4 diagnostics specified, all
runnable on existing post-exp1128 ensemble + ~100 synthetic examples.
Regime classification unambiguous. Topological-fencing risk
explicitly preserved as known unresolvable without full prototype.
**Date received:** 2026-05-02
**Source prompt:** `hmc-on-heterogeneous-energy-gradient-deep-think-prompt.md`

---

## TL;DR

| Component | Result |
|---|---|
| **Diagnostics** | 4 specified, all runnable on existing infrastructure |
| **Regimes** | A (HMC works), B (preconditioner needed), C (HMC inappropriate) |
| **Each regime** | Has clear empirical signature in first ~100 HMC steps |
| **Regime C fallbacks** | 3 candidates (Blocked Gibbs, Langevin, Surrogate) — each with its own diagnostic |
| **Self-correction** | Explicit lane check at end: no parameter prescriptions |
| **Unresolvable uncertainty** | Topological fencing — only diagnosable on long-horizon prototype runs |

---

## The 4 diagnostics (verbatim)

### Diagnostic 1: Symplectic Trajectory Reversibility

- **Tests:** detailed-balance preservation (severely threatened by finite-difference jump approximations)
- **Quantity:** sample `p_0`, simulate forward leapfrog L steps to `(x_end, p_end)`, negate to `-p_end`, simulate backward L steps to `(x_rev, p_rev)`, compute `||x_0 - x_rev||`
- **Direction:** **low** distance = compatible; **high** = discontinuous Z3/AST/JSON irreversibly break detailed balance
- **Caveat:** less reliable if step sizes too small to cross any boundary (artificial perfect reversibility, zero exploration)

### Diagnostic 2: Hamiltonian Energy Conservation (ΔH Drift)

- **Tests:** log-density smoothness (predicts viable proposal acceptance rate)
- **Quantity:** simulate multi-step leapfrog, compute `|ΔH| = |H(x_end, p_end) - H(x_0, p_0)|`. Track variance + frequency of exponential spikes
- **Direction:** **bounded, low variance** = compatible; **frequent unbounded spikes / NaNs** = fatal discontinuities from symbolic components
- **Caveat:** less reliable if examples drawn only from valid centers of AST/JSON basins, avoiding constraint boundaries

### Diagnostic 3: Cross-Component Gradient Norm Disparity

- **Tests:** Lipschitz boundedness uniformity, necessity of preconditioning
- **Quantity:** for each of 5 components, compute `Var(||w_i · ∇E_i(x)||)`. Compute ratio of max-component-variance to min-component-variance
- **Direction:** **near unity** = isotropic, well-conditioned (Regime A); **persistently high (orders of magnitude)** = severe anisotropy needs preconditioning (Regime B)
- **Caveat:** less reliable near structural local minima where smooth components vanish to zero (artificially inflates ratio)

### Diagnostic 4: Continuous Subspace Recovery (Post-Hoc Ablation)

- **Tests:** isolates whether discrete/symbolic components are the singular cause of leapfrog failure
- **Quantity:** offline, simulate leapfrog using **only** continuous gradient outputs (`w_Sem ∇E_Sem + w_PRM ∇E_ThinkPRM`); compute `|ΔH_continuous|`
- **Direction:** if `|ΔH_continuous|` bounded **while full `|ΔH|` from D2 explodes** → continuous substrate is HMC-compatible, discrete components are the strict bottleneck
- **Caveat:** highly reliable for localizing failure mode (provided continuous landscape isn't entirely flat)

---

## Regime classification (verbatim)

### Regime A — HMC works directly

- **Combination:** D1 low, D2 bounded, D3 low (near unity)
- **Empirical signature in first 100 steps:**
  - Dual-averaging adaptive step-size stabilizes quickly at macroscopic (non-vanishing) level
  - Acceptance rate oscillates around healthy positive equilibrium
  - Trace plots show rapid exploration; rapidly decaying autocorrelation
  - No "divergent transition" warnings in sampler logs

### Regime B — HMC works with per-component preconditioning

- **Combination:** D1 low, D2 bounded, **D3 HIGH (orders of magnitude disparity)**
- **Empirical signature:**
  - Without preconditioning: adaptive step-size adapter shrinks dramatically to survive steepest finite-difference gradients
  - Acceptance rate acceptable but sampler barely moves
  - Sluggish random walk; extreme autocorrelation; low ESS along continuous semantic dimensions
  - NUTS tree depth frequently maxes out

- **Preconditioning principle (verbatim):**
  > A "good preconditioner" aligns the mass matrix `M` with the expected inverse covariance of the aggregated gradients, scaling the momentum to equalize expected leapfrog displacement across highly disparate component scales. It fundamentally *solves* the multi-scale problem by ensuring a single unified time-step efficiently contours smooth basins while navigating steep finite-difference jumps safely.
  >
  > A preconditioner that strictly *masks* the problem artificially dampens or zeroes the inverse-mass along dimensions most sensitive to discrete structural constraints; this forces numerical integration to succeed but fundamentally ignores the verifiers, yielding generated states that rapidly drift into systematically violating Z3 logic and JSON schema.

  **The good/bad distinction:** preconditioner aligned with inverse covariance vs. preconditioner that suppresses sensitivity to symbolic constraints.

### Regime C — HMC fundamentally inappropriate

- **Combination:** D1 high, D2 massive exponential spikes, **but D4 stable** (continuous sub-ensemble works)
- **Empirical signature:**
  - Immediate cascade of "divergent transition" warnings
  - Acceptance rate crashes to ~0
  - Step-size adapter plummets to machine epsilon
  - Chain trace flatlines at initialization

- **Recommended fallbacks (each with its own diagnostic):**

  | Fallback | When to use | Diagnostic |
  |---|---|---|
  | **Blocked Gibbs / Metropolis-within-Gibbs** | D4 strictly passes | Continuous components via HMC; symbolic edits via separate Metropolis steps. Mathematically aligned. |
  | **Langevin with adaptive step** | L=1 vs L>1 comparison: single-step ΔH acceptable but multi-step accumulates fatal errors | Memoryless, momentum-free brakes gracefully at structural cliffs |
  | **Surrogate-gradient HMC** | Linear probe on continuous `E_Sem` hidden states predicts discrete gradients with high out-of-sample R² | Substitute true discontinuous gradients with smooth autograd surrogate |

---

## Unresolvable uncertainty (verbatim)

> **Global Ergodicity Breakdown (Topological Fencing).** Even if the diagnostics strongly classify the 100 synthetic examples into Regime A or B (local HMC compatibility), they cannot geometrically determine whether the globally valid Z3/AST/JSON regions form a broadly connected manifold, or a fragmented "Swiss cheese" landscape partitioned by impassable, infinite-energy constraint boundaries. If valid hypotheses are structurally disconnected, HMC will suffer from severe pseudo-ergodicity — mixing perfectly within a single reasoning basin, but suffering infinite autocorrelation when attempting to traverse invalid boundaries separating distinct modes. This macro-level topological connectivity dictates global Phase-4 ARC-AGI-3 manifold coverage and strictly cannot be diagnosed without running long-horizon chains on the full prototype to observe true inter-mode transition rates.

**Operational implication:** the diagnostics tell us about local HMC behavior. **Global ergodicity is a separate empirical question** — only answerable by running long-horizon chains on the actual prototype.

---

## Drift check

**Self-correction at end of response (verbatim):**

> *(Self-Correction / Lane Check: Per the `feedback_carnot_prediction_pattern.md` constraints, I have strictly avoided prescribing specific numerical parameters [e.g., exact step sizes `ε`, leapfrog step counts `L`, mass matrix diagonal recipes, or acceptance rate targets]. All thresholding is purely directional [high/low] and preconditioning is defined via mathematical objective [inverse covariance alignment] rather than numerical parameterization. I have maintained focus exclusively on diagnostic regime-classification and structural analysis.)*

**Drift assessment:**

- ✅ All 4 diagnostics specified by formula, not by threshold value
- ✅ Regime classifications use directional combinations (low/high) only
- ✅ Preconditioning principle stated as objective (inverse covariance), not as recipe
- ✅ Fallback diagnostics use directional comparisons (R² high, single-step vs multi-step error)
- ✅ Unresolvable uncertainty named explicitly rather than papered over

**This is the cleanest cross-validation-aware response of any Deep Think round to date.** Self-correction is explicit and itemized; no parameter prescriptions appear anywhere.

---

## Synthesis for the .91 prototype

The Phase-4 prototype now has a **two-task structure**:

### Task A: HMC Compatibility Diagnostics

**`exp11XX-hmc-compatibility-diagnostics`** (NEW, must run before sampler implementation)

Implement the 4 diagnostics as a Python module operating on the
existing post-exp1128 k=5 ensemble. Run on a 100-example synthetic
test corpus that includes both safe valid examples AND constraint-
boundary examples (per Diagnostic 2's caveat).

Acceptance: regime classification with documented empirical signature
data. ~3-5 days of focused work; no GPU required for the diagnostics
themselves.

### Task B: Sampler Implementation (regime-conditional)

The sampler design depends entirely on Task A's verdict:

```
Regime A → exp11XX-hmc-sampler-vanilla
           Use NumPyro HMC primitive with default leapfrog + adaptive
           step-size. ~5-10 days.

Regime B → exp11XX-hmc-sampler-with-preconditioner
           NumPyro HMC + per-component mass matrix aligned with
           inverse covariance. Verify the preconditioner *solves*
           rather than *masks* via post-hoc constraint-violation
           rate check on samples. ~7-12 days.

Regime C → exp11XX-fallback-sampler
           Three candidates by D4 / single-step / probe-R² combination:
           - Blocked Gibbs (continuous + symbolic separately)
           - Langevin with adaptive step
           - Surrogate-gradient HMC
           ~10-15 days, depending on which fallback wins.
```

### Task C: Topological Fencing Detection (later)

Once Task B's sampler runs, monitor for the unresolvable uncertainty
flag — long-horizon chains showing inter-mode transition rates near
zero. If observed: file `exp11XX-topological-fencing-mitigation` for
.92+ (e.g., parallel tempering across modes, mode-jumping moves).

---

## Operational recommendations

### Update the .90 planner pickup

The previously-filed `exp11XX-hmc-sampler-on-carnot-ebm` task should
be **split** into the diagnostic task + the regime-conditional sampler
task. The diagnostic task is **strictly cheaper** (no GPU, ~3-5 days)
and **strictly more informative** (regime classification before
implementation commits).

### Decision-changing properties

This response is decision-changing for .90/.91 in three ways:

1. **Don't build HMC blindly.** Run diagnostics first. The 3-5 day
   cost is recovered if Regime C is detected (avoids a wasted
   2-week HMC implementation that would never converge).

2. **Preconditioning has a real principle.** "Align with inverse
   covariance of aggregated gradients" is a concrete construction
   target, not a hand-tuning exercise. The good/bad distinction
   (solving vs. masking) gives an objective check.

3. **Topological fencing is a real risk** — and explicitly cannot
   be ruled out by the diagnostics. Plan the .92 mitigation task in
   advance so it's ready when needed.

### Cross-validation status — 6 of 6 Deep Think rounds clean

```
Q1 Energy inversion             None       Hypothesis A confirmed
Q2 Phase-3 attacks              None       7 attacks specified
Q3 GRPO + SP-IWPER              None       Outcome C, hybrid mandatory
Q4 Verifier degradation         Self-flagged (1)   Hybrid Structural Default
Q5 Phase-3 substrate            Self-flagged (2)   Option B held-out
Q7 HMC compatibility            Self-corrected     3 regimes with fallbacks
```

Q7's response includes explicit lane-check self-correction at the
end — the prediction-error pattern is now fully internalized across
the deep-research conversation.

---

## Recommended next steps

1. ✅ **Save this response** (done).
2. **Update `ops/known-issues.md`**: split `exp11XX-hmc-sampler-on-carnot-ebm`
   into Task A (diagnostics) + Task B (regime-conditional sampler).
   Task A becomes HIGHEST priority for .90; Task B's specific form
   is determined by Task A's outcome.
3. **When Q8 response arrives**: integrate with this Q7 response
   to define the complete Phase-4 prototype scope (sampler + action
   representation).
4. **In .91 prototype**: implement diagnostics first, classify
   regime, then implement matching sampler. The 3-5 day diagnostic
   cost is a forward-looking insurance policy.

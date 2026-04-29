# Deep Think Results — Round-12 Re-Derivation Under Proper Normalization

**Date received:** 2026-04-29
**Pairs with prompt:** `round12-renormalization-deep-think-prompt.md`
**Headline:** Round-12 qualitative claim **survives**. The position
paper's centerpiece result holds. But the gap-driver is **cumulative
rejection rate**, not verifier noise, and three engineering
prescriptions need revision.

---

## Executive Summary

Under proper normalization with $Z_t \in (0, 1)$:

- The asymptotic residual is $\delta_\infty^{\text{normalized}} = C_Z \cdot \|\nu_0^\perp\|$
  with $C_Z > 1$ a finite constant.
- $C_Z = \prod_t Z_t^{-1}$ converges because $Z_t \to 1$ as the active
  component depletes.
- The renormalization gap $\Delta_{\text{renorm}} = (C_Z - 1)\|\nu_0^\perp\|$
  is **strictly positive** and **driven by the cumulative rejection
  rate during early steps**, not by verifier accuracy $\varepsilon^*$
  or geometry as Carnot pre-registered.
- The Round-12 qualitative claim — "Carnot saturates the Fano lower
  bound modulo the joint null space" — **survives**. Floor is raised
  by the finite scalar $C_Z$, but saturation still holds.

**Engineering implications:** three prescriptions that Carnot
pre-registered as "unchanged" must change:

1. **Rejection-rate cap** is advisable (Carnot predicted not needed).
2. **Optimal trajectory is curriculum-style** (lenient → strict
   verifiers), not constant rejection (Carnot predicted optimum at
   $Z_t \to 1$).
3. **Rotation defence must be adjusted** to penalize verifiers that
   cause low $Z_t$, not just maximize orthogonality (Carnot predicted
   prescription unchanged).

## Cross-validation scorecard

| Q | Carnot prediction | Confidence | Deep Think | Verdict |
|---|---|---|---|---|
| Q1(a) decomposition | "Likely correct, modulo $\mu_P$ projection" | MEDIUM | Confirmed: $\nu_{t+1}^\perp = (\nu_t^\perp + (1-Z_t)\mu_P^\perp)/Z_t$, $\beta_t = 1/Z_t$ | ✅ **DIRECTIONALLY CORRECT** |
| Q1(b) $\alpha_t \beta_t < 1$ | "Should hold under rotation defence" | MEDIUM | NOT globally guaranteed; transient total residual inflation possible when $Z_t$ small | ⚠️ **PARTIALLY WRONG** |
| Q2(a) closed form | "$\prod Z_t^{-1}$ converges, $\delta_\infty = \delta_0 \cdot C, C \geq 1$" | MEDIUM | Confirmed: $\delta_\infty^{\text{normalized}} = C_Z \|\nu_0^\perp\|$, $C_Z > 1$ finite | ✅ **CORRECT** |
| **Q2(b) gap scaling** | **"$O(\varepsilon^*)$ — small in high-quality regime"** | MEDIUM | **Driven by cumulative rejection rate during early stages, NOT $\varepsilon^*$ or $\theta_F$** | ⚠️ **WRONG (driver attribution)** |
| Q2(c) monotone | "Can temporarily increase early; LOW confidence" | LOW | Confirmed: temporary increase when $1-Z_t$ large early | ✅ **CORRECT** (LOW-confidence prediction validated) |
| Q3(a) Round-12 survives | **"Qualitative claim SURVIVES"** | **HIGH** | **Confirmed: survives, requires quantitative correction** | ✅ **CORRECT (HIGH-confidence vindicated)** |
| Q3(b) gap driver | "$\varepsilon^*$ (verifier noise)" | MEDIUM | Rejection rate $Z_t$ in initial steps, not asymptotic geometry | ⚠️ **WRONG (driver attribution)** |
| Q4 theorem statement | "Up to renormalization factor $C(\varepsilon^*, \theta_F)$..." | MEDIUM | "...up to a finite scalar inflation penalty entirely determined by the sequence's cumulative rejection rate." | ⚠️ **DIRECTIONALLY CORRECT, WRONG ATTRIBUTION** |
| **Q5(a) rejection cap** | **"Probably NOT needed"** | MEDIUM | **Capping is advisable to bound $\Delta_{\text{renorm}}$** | ⚠️ **WRONG** |
| Q5(b) optimal $Z_t$ | "Optimum near $Z_t \to 1$" | LOW | Curriculum: lenient verifiers early → strict late | ⚠️ **DIRECTIONALLY DIFFERENT** |
| **Q5(c) rotation prescription** | **"Likely UNCHANGED"** | MEDIUM | **MUST be adjusted: penalize verifiers causing low $Z_t$** | ⚠️ **WRONG** |

**Score: 4 correct, 1 directionally correct, 6 wrong/partial.**

## The four critical findings

### 1. Round-12 survives at quantitative correction (Q3a)

The headline qualitative claim — "Carnot saturates the Fano lower
bound modulo joint null space" — **holds**. The position paper does
not need to retract Round-12.

The corrected statement (Q4 result):

> *"Under sequential verifier-filtered distillation, the model's
> residual error converges to the joint null space of the verifier
> suite, saturating the information-theoretic lower bound up to a
> finite scalar inflation penalty entirely determined by the
> sequence's cumulative rejection rate."*

This is the position-paper-ready theorem statement.

### 2. The gap-driver is cumulative rejection rate, not verifier noise (Q2b, Q3b)

Carnot's pre-registered intuition (multiple Qs) was that the
renormalization effect is dominated by verifier noise $\varepsilon^*$.
**Wrong.** Deep Think shows the gap is dominated by *cumulative
rejection rate during early steps*, when the active component still
has substantial mass and high rejection inflates the null component.

This is a different operating regime than Carnot assumed:
- High-quality verifier ($\varepsilon^* \to 0$) does NOT eliminate
  the gap — high rejection still causes inflation
- Even imperfect verifiers can have small gap if early rejections
  are kept low

### 3. Rotation defence must be revised (Q5c)

Round-7's rotation defence said: "select verifier maximally orthogonal
to the current residual." Round-12 inherited this prescription.

**Round-13 + this round say: revise it.** The correct selection
objective is now:

> Maximize active-component projection **subject to** keeping $Z_t$
> above a threshold (avoid excessively low rejection).

This is a constrained optimization, not pure orthogonality
maximization. The position paper's Round-7 result needs an updated
prescription that integrates with this round's findings.

### 4. Curriculum trajectory is optimal (Q5b)

Carnot pre-registered "optimum near $Z_t \to 1$ (low rejection)."
**Wrong.** Deep Think says the optimum is **curriculum-style**: start
with lenient verifiers ($Z_t$ closer to 1, more candidates accepted)
and transition to strict verifiers ($Z_t$ smaller, more aggressive
filtering) as the active component depletes.

This implies an additional Phase-3 deployment hyperparameter:
**verifier-strictness curriculum**. The conductor's PPSEBM relay or
the Phase-4 UCM+DVS module would need to track the active-component
norm to schedule the transition.

## Updated position paper outline

After this round, Phase-3 has the following corrected and *new*
prescriptions:

| Component | Status |
|---|---|
| 6 base verifiers + AND-composition (k=15) | unchanged |
| Round-12 saturation theorem (qualitative) | **unchanged**, with quantitative refinement: $\delta_\infty^{\text{normalized}} = C_Z \|\nu_0^\perp\|$ |
| Rotation defence (Round-7) | **revised**: constrained orthogonality maximization with $Z_t$ floor |
| Verifier-strictness curriculum | **NEW**: lenient→strict transition tracked by active-norm |
| Rejection-rate cap during early steps | **NEW**: bound $1-Z_t \leq r^*$ to control $C_Z$ |
| In-loop tripwire (Round-9) | unchanged for spec-gaming threat; complemented by UCM+DVS for drift (Round-13) |

## Cross-round consistency

This round's findings **further narrow Round-12** but do NOT
contradict any prior round directly. The Round-12 narrowing chain is
now:

- Round-12 (original): "Carnot saturates the Fano lower bound."
- Round-13 narrowing: "Saturates the *static* lower bound under
  strict-monotone $\rho_t$."
- This round narrowing: "Saturates the static lower bound *up to a
  scalar penalty $C_Z$ from rejection rate*."

The qualitative result "Carnot is the optimal architecture for its
threat model" survives both narrowings, but the position paper now
needs three new caveats and engineering prescriptions: (1) the
$C_Z$ correction, (2) the rotation-defence revision, and (3) the
curriculum prescription.

## Carnot prediction errors — pattern analysis

This is the *second* round in a row where Carnot got HIGH-confidence
predictions wrong:

- Round-13: 2 of 4 HIGH-confidence wrong (Q2b asymptote, Q5b architectural fix)
- This round: 1 of 1 HIGH-confidence right (Q3a survives — phew)

Total HIGH-confidence vindication record across both rounds: 1 right,
2 wrong. The cross-validation discipline continues to be load-bearing.

Pattern of errors: Carnot's intuition is well-calibrated on
**qualitative survival of theorems** (Q3a HIGH was right), but
**poorly calibrated on driver attribution** (Q2b, Q3b, Q5c MEDIUM
predictions of "what causes the effect" or "what to do about it"
were all wrong). The position paper should:

- Trust Carnot's qualitative claims (survival, existence)
- Distrust Carnot's quantitative attributions (which constant
  drives which gap, which engineering tweak is needed)
- Always cross-validate quantitative claims with Deep Think before
  publication

## Items for follow-up

1. **Phase-4 UCM+DVS quality lower bound** (still open from prior
   priority list; now particularly relevant since Phase-4 must also
   account for the curriculum trajectory).
2. **AND-composition mixing time on FPGAs** (still open).
3. **Curriculum scheduling theorem.** Given the Q5(b) finding, what's
   the optimal curriculum schedule? Is there a closed form for
   "transition from lenient to strict at active-norm threshold $\eta$"?
4. **Rotation-defence revision spec.** What's the precise constrained
   objective for the new rotation rule?

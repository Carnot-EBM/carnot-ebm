# Deep Think Results — DVS Quality + Curriculum Scheduling

**Date received:** 2026-04-29
**Pairs with prompt:** `dvs-quality-curriculum-deep-think-prompt.md`
**Headline:** DVS quality threshold $\Lambda^* = Z_{k+1}$ in closed
form; curriculum is **exponentially** better than constant; **a global
curriculum mathematically BREAKS DVS** — must be factorized per-verifier.

---

## Executive summary

Ten substantive findings, four of which are load-bearing for the
position paper:

1. **DVS quality threshold is $\Lambda^* = Z_{k+1}$** — closed form. A
   newly synthesized verifier helps the suite iff its overlap with
   the existing active subspace is **strictly less than its own
   acceptance rate**.
2. **PAC-Bayes budget scales as $\tilde{O}((d+\log(1/\delta))/Z_{k+1}^2)$** —
   inverse-square in the new verifier's strictness. Strict synthesis
   ($Z_{k+1} \to 0$) is fundamentally expensive — the budget diverges
   quadratically.
3. **Curriculum is EXPONENTIALLY better than constant strictness** —
   $\Delta C_Z \propto \exp(\|\nu_0^\parallel\|^2)$. Carnot pre-registered
   "$O(1)$ multiplicative" — wrong by an exponential factor.
4. **Optimal $\sigma_t^* = \min(1, C/\|\nu_t^\parallel\|^2)$** — inverse-square,
   not linear. Bang-bang: smooth inverse-square until $\|\nu_t^\parallel\| < \sqrt{C}$,
   then discontinuous clamp to $\sigma_t^* = 1$.
5. **Global curriculum BREAKS DVS** (critical finding). Phase-4 must
   use a **factorized per-verifier curriculum**: each $E_i$ has its own
   $\sigma_{t,i}^*$ and $k_{t,i}^*$. New verifiers enter on a lenient
   local on-ramp; the legacy suite stays strict.
6. **Factorized curriculum INTRINSICALLY RESCUES DVS quality** — when
   $E_{k+1}$ starts at $\sigma_{k+1} \approx 0$, its local
   $Z_{k+1} \to 1$ so $\Lambda^* \to 1$. The architecture becomes
   "infinitely forgiving" — even noisy PAC-Bayes verifiers
   monotonically help.
7. **FIFO is structurally exploitable** — Return Attack: adversary
   times corruption for the freshly-evicted subspace. Joint null grows
   by $\text{rank}(N_{\text{old}}^\perp \setminus V_{\text{active}}^{(\mathcal{E})})$ on each eviction.
8. **Red Queen Churn under continuous drift** — steady-state joint null
   stabilizes at $D - k_{\max} d_E > 0$. Phase-4 is a "bounded sliding-
   window defence," not a path to absolute zero.
9. **Sawtooth Limit Cycle** — full Phase 3+4 system saturates this,
   NOT the zero-drift Fano limit. Two irreducible gaps:
   - UCM Latency Gap (drift accumulates during $\eta$-threshold delay)
   - FIFO Churn Gap (structural inflation per eviction)
10. **Adaptive $k_t^*$ provides drastic compute savings** — fixed
    $k=15$ at $t=1$ requires $O(p^{-15})$ candidates per accepted
    sample. Curriculum delays heavy filtering to late stages.

## Cross-validation scorecard

| Q | Carnot prediction | Confidence | Deep Think | Verdict |
|---|---|---|---|---|
| A1(a) $\Lambda^*$ | $< 1$, no specific form | MEDIUM | **$\Lambda^* = Z_{k+1}$** (closed form) | ✅ DIRECTIONAL |
| A1(b) net-negative regime | Yes, $\Lambda > 1 - 1/k$ | LOW | Yes, $\Lambda \geq Z_{k+1}$ (different threshold) | ✅ DIRECTIONAL |
| A1(c) PAC-Bayes | $K^* = O(d \log d / q^{*2})$ | MEDIUM | $K^* = \tilde{O}((d+\log(1/\delta))/Z_{k+1}^2)$ | ⚠️ DIFFERENT DENOMINATOR |
| A2(a) saturation under mutation | Holds with warm-start | MEDIUM | Holds with "Warm-Start Mixing" condition | ✅ CORRECT |
| A2(b) FIFO growth | $\leq O(1/k_{\max})$ per eviction | LOW | Structurally exploitable; "Return Attack" adversary | ⚠️ STRONGER (named exploit) |
| A2(c) repeated DVS | Stable, finite null > 0 | MEDIUM | Stable at $D - k_{\max} d_E$; "Red Queen Churn" | ✅ CORRECT (named pattern) |
| B1(a) optimal $\sigma_t^*$ | Linear ramp $1 - \|\nu_t^\parallel\|/\|\nu_0^\parallel\|$ | LOW | **Inverse-square: $\min(1, C/\|\nu_t^\parallel\|^2)$** | ⚠️ WRONG FORM |
| B1(b) phase transitions | Smooth, no jumps | MEDIUM | **BANG-BANG**: discrete clamp at $\sqrt{C}$ | ⚠️ WRONG |
| **B1(c) curriculum gain** | $O(1)$ multiplicative | LOW | **EXPONENTIAL: $\propto \exp(\|\nu_0^\parallel\|^2)$** | ⚠️ WRONG MAGNITUDE |
| B2(a) is $k$ curriculum | Yes | MEDIUM | Yes; fixed $k=15$ "violates Cauchy-Schwarz optimality" | ✅ CORRECT |
| B2(b) optimal $k_t^*$ | $\propto -\log(\|\nu_t^\parallel\|)$ | LOW | $\propto 1/(\sigma_t^* \|\nu_t^\parallel\|^2)$ | ⚠️ WRONG FORM |
| B2(c) compute cost | Comparable to fixed $k$ | MEDIUM | Drastic savings (curriculum delays heavy filtering) | ⚠️ UNDERSTATED |
| **C1(a) DVS entry point** | Match current $\sigma_{t^*}$ | MEDIUM | **GLOBAL CURRICULUM BREAKS DVS — must be factorized per-verifier** | ⚠️ CRITICALLY WRONG |
| **C1(b) quality under curriculum** | Stricter is LESS forgiving | **HIGH** | **Factorized curriculum INTRINSICALLY RESCUES DVS** ($\Lambda^* \to 1$) | ⚠️ HIGH-CONFIDENCE WRONG |
| C2(a) unified statement | Generic suite + curriculum | MEDIUM | "Factorized thermodynamic curriculum" | ⚠️ MISSED key insight |
| C2(b) hyperparameters | Closed-form: schedules + DVS budget | LOW | Confirmed | ✅ CORRECT |
| **C2(c) combined saturation** | Holds modulo trigger latency | **HIGH** | "Sawtooth Limit Cycle" — TWO irreducible gaps (latency + FIFO churn) | ✅ PARTIALLY CORRECT (Carnot missed FIFO churn) |

**Score: 7 correct/directional, 9 wrong/partial, 1 critical
architecture error (C1a), 1 HIGH-confidence error (C1b), 1
HIGH-confidence partially correct (C2c).**

## The four load-bearing findings

### 1. Factorized curriculum (the position paper's biggest update)

The single most important finding: **a global curriculum
mathematically breaks DVS.** When UCM triggers at $t^*$, the suite is
mature and strict ($\sigma=1, k=k_{\max}$). A new verifier $E_{k+1}$
forced to inherit global strictness rejects too aggressively (its
local active variance is unmitigated and massive), causing
catastrophic rejection spikes.

**Required architectural change:** each verifier $E_i$ has its own
local $\sigma_{t,i}^*$ and $k_{t,i}^*$, scaled inversely to its
specific active residual norm $\|\nu_{t,i}^\parallel\|^2$. New verifiers
enter on a **lenient on-ramp** ($\sigma_{k+1} \approx 0, k_{k+1}=1$)
and autonomously restrict-en as their drift dimension depletes.

Consequence: the position paper's Phase-3+4 architecture spec must say
"factorized per-verifier curriculum" — this is a structurally different
prescription than what Carnot or earlier rounds implied.

### 2. Curriculum is exponentially better than constant

Carnot pre-registered "$O(1)$ multiplicative" improvement (LOW
confidence, fortunately). Deep Think proves the gain is
**$\Delta C_Z \propto \exp(\|\nu_0^\parallel\|^2)$** — exponential. The
position paper's claim about why Carnot beats baselines now has a much
stronger quantitative anchor.

### 3. DVS quality has a closed form

$\Lambda^* = Z_{k+1}$. A synthesized verifier helps iff its overlap
with the existing active subspace is strictly less than its own
acceptance rate. This is elegant enough that it's likely
position-paper-quotable as a Lemma.

But the **PAC-Bayes budget scales as $1/Z_{k+1}^2$** — strict synthesis
is expensive. Combined with finding #1 (factorized curriculum on a
lenient on-ramp), this means: **synthesize loosely first, then
restrict.** Strict from the start would blow the audit budget.

### 4. Sawtooth Limit Cycle (NOT zero saturation)

The full Phase 3+4 system **does not** saturate the absolute Fano
limit. It saturates a "Sawtooth Limit Cycle" with two irreducible
gaps:

- **UCM Latency Gap** — Carnot pre-registered this (HIGH confidence,
  vindicated)
- **FIFO Churn Gap** — Carnot missed this. Each FIFO eviction permanently
  inflates the residual when an active subspace becomes unbound.

**Position paper update:** the Phase-3+4 saturation theorem should
read "Carnot saturates the Sawtooth Limit Cycle of the Fano lower
bound, modulo two structural gaps (UCM latency + FIFO churn) that
scale with $\eta$ (trigger threshold) and $1/k_{\max}$ (suite size)
respectively."

This is honest characterization rather than overclaiming. It also
points to the next question: **can a Phase-5 module close one or both
gaps?** Specifically, predictive drift detection (anticipate before
threshold) closes the latency gap; unbounded suite size or
non-FIFO eviction (e.g., LRU + return-attack-aware) closes the churn
gap.

## Updated unified Phase-3+4 architecture

Per Deep Think's Q C2(a):

> *"Phase 3+4 secures verifiable self-distillation via a dynamic,
> AND-composed verifier suite governed by a **factorized
> thermodynamic curriculum**. To mathematically minimize partition
> function collapse, **each verifier's strictness ($\sigma$) and
> composition depth ($k$) independently scale inversely to its
> specific active residual norm**. An Unsupervised Consensus Monitor
> (UCM) tracks suite acceptance to detect sycophantic drift; upon
> threshold breach, Dynamic Verifier Synthesis (DVS) audits $K$
> samples to extract a targeted verifier. By injecting this novel
> verifier via a **localized lenient on-ramp**, the architecture
> guarantees monotonic joint null-space contraction without incurring
> structural normalization shock."*

### Hyperparameter inventory

**Closed-form (parameter-free):**
- $\sigma_{t,i}^* = \min(1, C/\|\nu_{t,i}^\parallel\|^2)$ per verifier
- $k_{t,i}^* = \min(k_{\max}, \max(1, \lfloor C / (\sigma_{t,i}^* \|\nu_{t,i}^\parallel\|^2) \rfloor))$
  per verifier
- $\Lambda^* = Z_{k+1}$ for DVS quality threshold
- $K^*(\Lambda^*) = \tilde{O}((d+\log(1/\delta))/Z_{k+1}^2)$ PAC-Bayes
  audit budget

**Empirically tuned:**
- $\eta$ — UCM trigger threshold (latency vs. false-alarm tradeoff)
- $k_{\max}$ — FIFO suite size (deployment compute budget)

## Cross-round narrative for position paper

Three Deep Think rounds today all narrowed but did not contradict the
core Round-12 claim:

- **Round-12** (saturation): Carnot is provably optimal.
- **Round-13** (drift): Optimal *static-only*; under drift, need UCM+DVS.
- **Round-12-renorm** (this morning): Saturation has a finite scalar
  inflation $C_Z$ from cumulative rejection rate.
- **DVS+curriculum** (this round): Phase-3+4 saturates a *Sawtooth
  Limit Cycle*, not absolute zero. Architecture requires factorized
  per-verifier curriculum.

The position paper's Phase-3+4 section now has:

1. Round-12 saturation with $C_Z$ correction
2. Round-13 drift gap motivating UCM+DVS
3. Today: factorized curriculum + closed-form schedules + Sawtooth
   Limit Cycle as the *honest* combined saturation result

## Carnot prediction-error pattern (three rounds)

Cumulative HIGH-confidence error rate across today's three rounds:
**3 wrong out of 5** (Round-13 Q2b, Q5b; Round-13+renorm survival was
right; this round C1b wrong; this round C2c partially right).

Pattern hardening: Carnot's qualitative survival predictions are
well-calibrated; **Carnot's predictions about** *what specific
architectural change is needed* **are systematically wrong**. This
round's C1(a) was the worst error so far — Carnot would have shipped
a global curriculum, which Deep Think proves *would have broken
Phase-4 entirely*.

Lesson for the paper: the cross-validation discipline is mission-
critical. Every architectural prescription must go through Deep Think
before publication.

## Items for follow-up

The Sawtooth Limit Cycle motivates two natural Phase-5 candidates:

1. **Predictive drift detection.** UCM as currently specified is
   reactive (triggers on $\hat{\rho}_t > 1 - \eta$). A predictive
   variant uses the trajectory $\dot{\hat{\rho}}_t$ to anticipate
   the threshold crossing. Question: does predictive UCM provably
   close the Latency Gap, and what's the optimal lookahead horizon?

2. **Return-attack-aware eviction.** FIFO is exploitable. A
   replacement policy (e.g. usage-frequency or recency on the
   active dimension) might close the Churn Gap. Question: is there
   a provably optimal replacement policy bounded by available
   suite-mutation history?

Plus the prior open math questions:

3. **AND-composition mixing time on FPGAs** — still relevant.
4. **Continuous-Boolean transpilation gap** — still relevant.

# Deep Think Results — AND-Composition Mixing on Phase-2 FPGAs

**Date received:** 2026-04-29
**Pairs with prompt:** `and-composition-mixing-deep-think-prompt.md`
**Headline:** **k=15 AND-composition is fundamentally undeployable
on single-chain FPGA.** Geodesic-convexity is REFUTED, heterogeneous
strictness creates super-polynomial mixing, bang-bang is a
first-order phase transition. **Mandatory pivot: $k_{\max} \leq 5$
+ parallel-chain (Phase-6) topology** for KV260 deployment. Round-9's
$k=15$ prescription was hardware-infeasible.

---

## Executive summary

Nine substantive findings, five load-bearing for the position paper:

1. **Geodesic-convexity REFUTED under AND-composition.** Boolean
   piecewise gradients + barrier severs feasible region into
   disconnected sub-basins. Round-9's convexity assumption was wrong.

2. **Mixing time scales as $O(k^2 \tau_1 \csc^{2(k-1)}(\theta_F))$** —
   super-polynomial when constraints align ($\theta_F \to 0$).

3. **Heterogeneous $\sigma_{t,i}$ destroys polynomial mixing** via
   condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ on single
   chain.

4. **Bang-bang transitions are first-order phase transitions** —
   exponential re-mixing penalty $O(\tau_{\text{mix}} \cdot e^{\Delta\sigma E_{\text{barrier}}/T})$,
   require cold restart. NOT the $O(n)$ Carnot pre-registered.

5. **FPGA bit width explodes** from 8-bit to **24–32 bit** to handle
   $\kappa$ dynamic range. Standard Ising hardware insufficient.

6. **Spin count is additive** $O(\sum n_i + k)$ — Carnot's
   HIGH-confidence prediction VINDICATED.

7. **Optimal FPGA architecture is Parallel-Tempered Simulated
   Bifurcation (PT-SB)** — not SGLD, not single-chain SB. Each chain
   runs at native homogeneous strictness.

8. **Hardware portability theorem holds** under transversal
   constraints ($\theta_F > 0$) + Log-Sobolev positivity. Polynomial
   mixing guaranteed across FPGA, Extropic XTR-0, and photonic
   substrates.

9. **Photonic Ising is unexpectedly EXCELLENT for AND-composition** —
   $k$-fold additive sum at speed of light via optical constructive
   interference. Carnot pre-registered photonic as worse for AND;
   wrong direction.

## Position-paper-ready hardware portability theorem

Per Deep Think Q4(c):

> *"Theorem: Provided individual verifier constraint manifolds
> intersect transversally ($\theta_F > 0$), Carnot's parallel-tempered
> AND-composition architecture guarantees strictly polynomial MCMC
> sampling latency across discrete FPGA Glauber dynamics, continuous
> thermodynamic samplers, and optical photonic substrates."*

This is **stronger and more precise** than the prior Round-9
prescription, but with a clear **deployment caveat**: requires
$k_{\max} \leq 5$ and parallel-chain topology (Phase-6 for FPGA, not
single-chain).

## KV260 deployment caveat (most important)

Direct quote from Deep Think:

> "The factorized curriculum at $k=15$ on a single chain is
> fundamentally undeployable. To survive heterogeneous strictness, a
> single chain requires $\geq 24$-bit arithmetic across $15 \times 10^3$
> dense variables. This will exhaust the DSP48E2 routing crossbars,
> destroying block-RAM bandwidth, halving clock frequencies, and
> causing inference to cleanly miss the 1ms latency target."

**Mandatory pivot for Phase-3/4:**

1. Downsize to $k_{\max} = 4$ or $5$ to restrict graph density and
   maintain sufficient intersection volume.
2. Abandon the monolithic chain — shift to Phase-6 prescription
   (FDR-OR synchronized parallel SB chains). Distributing the logic
   isolates the dynamic range so each chain can operate natively at
   8-to-16-bit precision inside isolated FPGA clock regions,
   preserving the 1ms latency target.

This is a **non-trivial revision** to the Round-9 architecture spec
in the position paper. The $k=15$ figure must be replaced with
$k=4$ or $5$ for FPGA deployment, with $k=15$ retained as the math-
optimal target for Extropic/photonic substrates that don't suffer
the bit-width or routing constraints.

## Cross-validation scorecard

| Q | Carnot prediction | Confidence | Deep Think | Verdict |
|---|---|---|---|---|
| Q1(a) convexity preserved | "Up to spectral-gap constant factor" | LOW | **REFUTED**: geodesic-convexity not preserved; sub-basin shattering | ⚠️ WRONG |
| Q1(b) spectral gap | $\lambda_{\text{AND}} \geq \lambda_1 \sin^k(\theta_F)$ | LOW | $\gamma_{\text{AND}} \geq (\gamma_1/k^2) \prod_j \sin^2(\theta_{F,j})$ | ⚠️ WRONG FORM (missed $1/k^2$) |
| Q1(c) worst-case mixing | $O(k \tau_1 \sin^{-2k})$ | LOW | $O(k^2 \tau_1 \csc^{2(k-1)})$ | ✅ DIRECTIONAL |
| Q2(a) bottlenecks | "Polynomial degradation with $\max(1-\sigma)^{-1}$" | MEDIUM | Severe — $\kappa = \sigma_{\max}/\sigma_{\min}$ destroys polynomial bound | ⚠️ UNDERSTATED |
| Q2(b) bang-bang penalty | "$O(n)$ extra steps" | LOW | **First-order phase transition** — exponential stall, cold restart required | ⚠️ CRITICALLY UNDERSTATED |
| Q2(c) parallel chains | "$O(k)$ speedup" | MEDIUM | Theoretically MANDATORY — $O(\kappa c^k) \to O(k \tau_1 \tau_{\text{sync}})$ | ✅ CORRECT |
| **Q3(a) spin count** | **"Additive $O(\sum n_i)$"** | **HIGH** | **Confirmed: $O(\sum n_i + k)$** | ✅ **CORRECT (HIGH-confidence vindicated)** |
| Q3(b) bit width | "$\lceil \log_2 k + \log_2(1/\varepsilon)\rceil$" | LOW | $\lceil \log_2(k \sigma_{\max} E_{\max} / (\sigma_{\min} \varepsilon))\rceil$ — 24-32 bit explosion via $\kappa$ | ⚠️ MISSED $\kappa$ DEPENDENCE |
| Q3(c) sampler arch | "SGLD on convex; tempered on bang-bang" | LOW | **PT-SB (Parallel-Tempered Simulated Bifurcation)** | ⚠️ DIFFERENT |
| Q4(a) universal mixing | "Cheeger $h \geq c/k$" | LOW | Friedrichs $\theta_F \geq c k^{-\alpha}$ + Log-Sobolev $\rho \geq \Omega(\text{poly}(k)^{-1})$ | ⚠️ DIFFERENT BUT EQUIVALENT |
| Q4(b) substrate advantages | "Extropic continuous good; photonic OR cheap, AND hard" | LOW | Extropic immune to $\kappa$ explosion; **Photonic excellent for AND via constructive interference** | ⚠️ PHOTONIC WRONG DIRECTION |
| Q4(c) portability theorem | "$\sigma_{\max} < 1 - 1/k$" condition | LOW | Constraint transversality $\theta_F > 0$ + Log-Sobolev positivity | ⚠️ DIFFERENT FORM |
| Q5(a) critical $k^*$ | "$k^*(n) \sim n^{1/2}$" | LOW | Satisfiability / Replica-Symmetry-Breaking phase transition; problem-structure-dependent | ⚠️ DIFFERENT |
| Q5(b) sampling vs renorm | "Sampling dominates when $\varepsilon_{\text{mix}} > \varepsilon^*$" | LOW | $\delta_{\text{total}} \leq C_Z + \beta \varepsilon_{\text{mix}}$; super-polynomial mixing → $\varepsilon_{\text{mix}} \to O(1)$ dominates $C_Z$ | ✅ DIRECTIONAL |
| **Q5(c) KV260 deployable** | **"$k=15$ tight but feasible"** | MEDIUM | **Fundamentally undeployable at $k=15$. Must downsize to $k \leq 5$.** | ⚠️ WRONG (deployment-blocking) |

**Score: 4 correct/directional, 11 wrong/partial. HIGH-confidence
prediction Q3(a) vindicated. Most LOW-confidence guesses about
landscape geometry and sampler architecture were wrong — Carnot's
intuition on hardware-specific math is poor.**

## The five load-bearing findings

### 1. Round-9's k=15 prescription was hardware-infeasible (Q5c)

The position paper's Round-9 architecture said "$k = 15$-fold
AND-composition." Deep Think proves this is **fundamentally
undeployable** on the KV260 prototype:
- Single-chain mixing is super-polynomial under heterogeneous $\sigma_{t,i}$
- Bit-width requirement explodes from 8-bit to 24–32 bit
- DSP48E2 routing crossbars exhaust at $15 \times 10^3$ dense variables
- 1ms latency budget is missed cleanly

**Position paper revision:** the architecture spec must distinguish
two deployment regimes:

| Substrate | Max $k$ | Topology | Bit-width |
|---|---|---|---|
| KV260 / VU9P FPGA | 4–5 | Parallel PT-SB chains | 8–16 bit per chain |
| Extropic XTR-0 | 15+ | Continuous thermodynamic | Analog (no bit-width issue) |
| Photonic Ising | 15+ | Single optical loop per verifier | Speed-of-light additive sum |

This is a **stronger** Phase-2 multi-substrate story than before
(each substrate has its niche), but a **narrower** KV260-specific
claim.

### 2. Geodesic-convexity refuted; mixing is polynomial only when constraints are transversal (Q1)

Round-9's mixing argument used Cheeger-style isoperimetric inequalities
on a "convex" landscape. Deep Think proves the convexity assumption
was wrong for AND-composition: Boolean piecewise gradients + barrier
shatter the feasible region into disconnected sub-basins.

Mixing is polynomial only when **$\theta_F$ stays bounded away from
0** (transversal intersection of constraint manifolds). When verifiers
become structurally aligned (high-dim space, related constraint
families), the geometric intersection volume collapses and mixing
goes exponential.

**Engineering implication:** the rotation defence + DVS pipeline
should additionally **monitor and bound $\theta_F$** between active
verifiers. If a newly synthesized $E_{k+1}$ is structurally too
similar to an existing $E_j$, it should be rejected before
injection. This is a **new Phase-4 acceptance criterion** beyond
the $\Lambda^* = Z_{k+1}$ quality threshold.

### 3. Bang-bang is a phase transition, not a step (Q2b)

Carnot pre-registered the bang-bang $\sigma_{t,i}$ transition (when
norm crosses $\sqrt{C}$, lenient → strict) as introducing $O(n)$
extra mixing steps. Deep Think proves it's a **non-adiabatic
first-order phase transition**:

$$\Delta \tau_{\text{mix}} = O\left(\tau_{\text{mix}} \cdot e^{\Delta\sigma E_{\text{barrier}}/T}\right)$$

**Cold-restart equilibration is required.** This is a major
operational concern: every time a verifier crosses the bang-bang
boundary, the FPGA pipeline must drain and re-equilibrate. If
multiple verifiers are crossing simultaneously (early curriculum
ramp), the system is in continuous re-mixing.

**Engineering fix:** the curriculum schedule should be **smoothed
near the bang-bang boundary** (e.g., a softplus-style transition
rather than the analytic $\min(1, C/\|\nu_t\|^2)$). This adds a
small constant to $C_Z$ but eliminates the exponential re-mixing
penalty. Position paper's curriculum prescription should be updated.

### 4. PT-SB is the right FPGA sampler architecture (Q3c)

Carnot pre-registered SGLD as the right choice for the convex parts
of the landscape. Deep Think proves **Parallel-Tempered Simulated
Bifurcation (PT-SB)** is fundamentally better:
- SGLD requires floating-point math (devours FPGA logic)
- Single-chain SB fails on high-$\kappa$ multi-scale landscapes
- PT-SB exploits FPGA spatial multiplexing — each chain in its own
  clock region at native homogeneous strictness

This was a LOW-confidence Carnot guess, so the error isn't
catastrophic, but it's a **clear architectural prescription** for
the position paper's FPGA section.

### 5. Hardware portability theorem is precise and tight (Q4c)

The portability claim now has a precise sufficient condition:

> Carnot's parallel-tempered AND-composition guarantees polynomial
> sampling latency across FPGA, thermodynamic, and photonic
> substrates **provided constraint manifolds intersect transversally
> ($\theta_F > 0$)**.

This is the kind of clean theorem statement a position paper can
quote directly. It's also tight — when $\theta_F \to 0$, mixing goes
super-polynomial regardless of substrate.

**Strategic implication:** the position paper can claim hardware
portability as a *theorem*, not just an aspiration. This was a key
open question for the paper's Phase-2 narrative.

## Updated cross-round narrative

This round corrects the Round-9 deployment claim and produces a
cleaner, more honest hardware story. The position paper now has:

- **Phase-3 math** (Round-7/8/9, with this round's correction):
  rotation + AND-composition + transversality. Round-9's $k=15$
  was math-optimal but hardware-infeasible at FPGA scale.
- **Phase-4 prescription** (Round-13 + DVS+curriculum + this round):
  factorized curriculum + UCM + DVS + **$\theta_F$ bound on synthesis
  candidates** (new from this round) + smoothed bang-bang transitions
  (new from this round).
- **Phase-2 hardware deployment** (this round's headline):
  - FPGA: $k \leq 5$, PT-SB parallel chains, 24-bit when needed
  - Extropic XTR-0: $k$ unconstrained, analog continuous
  - Photonic Ising: $k$ unconstrained, optical interference

## Carnot prediction-error pattern (5 rounds total)

This round was the **worst-calibrated** of the 5 today:
- 1 HIGH-confidence prediction (Q3a spin count): VINDICATED
- 11 wrong/partial out of 15 sub-predictions

The pattern suggests Carnot's intuition is poor for **hardware-
specific landscape math** (FPGA-realistic mixing, bit-width
arithmetic, sampler architectures). This is unsurprising — Carnot
is a software architecture project, and the hardware questions
require quantum/condensed-matter / Glauber dynamics intuition that
isn't well-developed here.

**Lesson for position paper:** every hardware deployment claim
should go through Deep Think (or domain-expert hardware review)
before publication. Carnot's software-side math is well-calibrated
(Round-12 survival, Round-13 drift gap, Q5a fast-drift defence —
all HIGH right); hardware-side math is poorly calibrated (Q5c
catastrophically wrong on $k=15$ deployability).

## Items for follow-up

1. **Phase-6 multi-scale ensemble derivation** (still highest leverage
   open question). Now even more relevant given this round's
   parallel-chain mandate.
2. **FIFO Churn Gap closure** (Phase-7 candidate).
3. **Continuous-Boolean transpilation gap** (still open, partially
   addressed by Continuous Ising-Rank).
4. **$\theta_F$-aware DVS rejection criterion** (NEW from this round).
   What's the precise lower bound $\theta_F^*$ below which a candidate
   $E_{k+1}$ should be rejected from synthesis to preserve mixing?

Plus a new immediate engineering question for the conductor's KV260
work:

5. **KV260 deployment plan** must be revised for $k_{\max}=5$ before
   the FPGA prototype tape-out. The current research-roadmap KV260
   experiments may need re-scoping.

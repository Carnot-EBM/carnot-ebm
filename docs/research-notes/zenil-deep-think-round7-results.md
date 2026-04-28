# Deep Think Round-7 Results — Multi-Verifier Architecture is Mathematically Required

**Status:** Authoritative. Builds on Round-6 validation results.
**Date:** 2026-04-28 (late evening)
**Prompt:** `zenil-deep-think-round7-followup-prompt.md`

Deep Think delivered three deployable results plus one load-bearing
strategic finding: **multi-verifier pre-emptive rotation is the ONLY
mathematically rigorous defence against null-space mimicry attacks**
on verifier-filtered self-distillation. Statistical tests on verifier
moments cannot distinguish benign orthogonality stall from adversarial
specification gaming.

## Q1 — Closed-form δ_plateau (refined Carnot prediction)

$$
\boxed{\delta_{\text{plateau}} = \frac{\delta_0 \|\Pi_{E^\perp} g_0\|_{L^2(\mu_P)}}{\sqrt{1 - (\varepsilon / b_1)^2}}}
$$

where:
- $\delta_0 \|\Pi_{E^\perp} g_0\|$ is the absolute initial mass orthogonal
  to $E$'s eigenspace (the unmeasurable error component)
- $\varepsilon$ is the verifier's noise bound
- $b_1 = \langle \phi_1, E\rangle$ is the spectral overlap of the
  verifier energy with its slowest-decaying eigenmode

**Interpretation:**
- When $\varepsilon \ll b_1$: $\delta_{\text{plateau}} \approx \delta_0 \|\Pi_{E^\perp} g_0\|$
  — plateau is just the orthogonal mass, as expected.
- When $\varepsilon \to b_1$: denominator $\to 0$, plateau diverges —
  verifier accuracy becomes the bottleneck before orthogonality does.

**Phase 2 hardware sizing:** the formula gives concrete numbers.
Estimate $\|\Pi_{E^\perp} g_0\|$ at training start; measure $b_1$ from
energy spectrum; the plateau tells you when FPGA acceleration stops
helping.

## Q2 — Multi-verifier theory (mostly refuted Carnot predictions)

### (a) Coverage condition — REFINED

The qualitative null-space condition $\bigcap_i \mathrm{null}(\mathcal{K}_{E_i}) = \{0\}$
is **necessary but insufficient** in infinite-dimensional space. It
allows zero-minimum-angle subspace intersections, permitting infinitely
slow asymptotic stalls.

**Required quantitative condition (Friedrichs angle):**
$$
\sum_{i=1}^{k} \mathcal{K}_{E_i}^2 \geq \gamma \mathcal{I} \quad \text{on } L^2_0(\mu_P), \text{ with } \gamma > 0.
$$

This is a *strict joint spectral gap* — the $E_i$'s must have
quantitative coverage, not just topological coverage.

### (b) Optimal rotation cadence — REFUTED Carnot prediction

**Pre-emptive rotation is strictly optimal.** Carnot's prediction
("rotate only on stall") was wrong.

Reason: the instantaneous decay rate $\Delta\delta_t \approx -\Phi_i / T_{\text{floor}}$
is linear in $\Phi_i$. Staying on a single verifier causes $\Phi_i$ to
decay exponentially as the residual aligns with its null-space, plunging
into severe diminishing returns. Greedy max-Φ rotation at every round
"rides the upper envelope of available restorative forces."

### (c) Ensemble vs rotation — REFUTED Carnot prediction

**Rotation strictly dominates ensemble combine asymptotically.**

The math:
- **Rotation:** $N_t^{\text{rot}} > \frac{\mathrm{Var}(E_{i^*}) T \tau_{\text{int}}}{2 (\max_i \Phi_i) \delta_t}$
- **Ensemble:** $N_t^{\text{ens}} > \frac{\mathrm{Var}(\sum E_i) T \tau_{\text{int}}^{\text{ens}}}{2 (\sum_i \Phi_i) \delta_t}$

Two compounding effects make ensemble worse:
1. Variance scaling: $\mathrm{Var}(\sum E_i) \approx k \cdot \mathrm{Var}(E_{i^*})$
2. **Multi-modal glassy frustration:** combining constraint landscapes
   creates frustrated topology, so MCMC mixing time $\tau_{\text{int}}^{\text{ens}}$
   "blows up factorially."

Net: $N_t^{\text{ens}} \approx \mathcal{O}(k \cdot N_t^{\text{rot}})$,
plus the $\tau_{\text{int}}^{\text{ens}}$ blowup. Rotation is strictly
better.

**Counter-intuitive engineering result.** Ensembles are the natural
default for "more is better" intuitions, but for verifier-filtered
sampling, they create frustrated landscapes that break MCMC.

### (d) Minimum k — REFINED

$$
k_{\min} = \left\lceil \frac{D}{d} \right\rceil
$$

where $D$ is residual error space's intrinsic effective dimension and
$d$ is per-verifier active subspace dimension.

For reliable shrinkage to a target plateau:
$$
k \geq \frac{D}{d} \ln\!\left(\frac{\delta_0}{\delta_{\text{plateau}}^*}\right)
$$

— logarithmic-in-improvement-ratio for randomly-overlapping coverage.

## Q3 — Distinguishing orthogonality from gaming (the load-bearing finding)

### (a) Higher-order statistics

- **Orthogonality stall ($H_0$):** $g_t$ supported on level-set variations
  of $E$. All cross-moments vanish: $\mathrm{Cov}_{\mu_P}(g_t, E^m) \to 0$
  for all $m \geq 1$.
- **Adversarial gaming ($H_1$):** model mode-collapses onto OOD
  $E$-blind-spots. Distorts local variance of $E$. Specifically,
  $\mathrm{Cov}_{\mu_P}(g_t, E^2) \ll 0$.

### (b) Discriminating hypothesis test

$$
\mathcal{T} = \widehat{\mathbb{E}}_{Q_t}[E^2] - \mathbb{E}_{\mu_P}[E^2]
$$

Under $H_0$: $\mathcal{T} \sim \mathcal{N}(0, \mathrm{Var}_{\mu_P}(E^2)/N_t)$.

Reject $H_0$ if $\mathcal{T} < -z_\alpha \sqrt{\mathrm{Var}_{\mu_P}(E^2)/N_t}$.

**Power:** near-perfect against continuous reward-model hacking via
variance suppression.

### (c) Carnot's structural protection — partial

Boolean $E$ (Z3 satisfaction, $E \in \{0, 1\}$) means $E^2 = E$
universally. This **deletes independent higher-order moments** —
neutralising the Q3 test.

Adversary cannot exploit *continuous* false-positives (learned PRM
hallucinations don't apply). But adversary can still execute
**Specification Gaming** — generate semantically empty trivial outputs
(`return 0;`) that satisfy $E = 0$ identically while failing the
problem's actual intent.

## Final integration — THE WEAKEST LINK

**The Q3 test has ZERO power against an attack that exactly mimics Q1.**

When the adversary specification-games with $E = 0$ trivial outputs:
- $\widehat{\mathbb{E}}_{Q_t}[E^2] = 0$ (model outputs all valid, $E^2 = E = 0$).
- $\mathbb{E}_{\mu_P}[E^2] = 0$ (truth distribution is also all valid).
- $\mathcal{T} = 0 - 0 = 0$.

The adversarial Goodharting **perfectly embeds itself within the
mathematical null-space of $E$**, exactly mimicking the orthogonality
stall signature.

Statistical tests on a single verifier's moments cannot distinguish.

### THE ONLY DEPLOYABLE SAFETY NET

**Multi-verifier pre-emptive rotation is mathematically required.**
Not optional, not nice-to-have — *the only architecture that defends
against null-space mimicry*.

If the model specification-games $E_1$ (safety checker), pre-emptively
rotating to $E_2$ (liveness property OR semantic AST density check)
breaks the mimicry instantaneously: the trivial code catastrophically
fails $E_2$'s topological requirements, $\widehat\Phi_2$ spikes
violently, and the illusion of stall shatters.

The safety guarantee requires **semantically orthogonal** verifiers —
$E_2$ must measure something $E_1$ cannot game by construction.

## Implications for Carnot

### Architecture (Phase 3 foundation model)

Phase 3 self-distillation MUST include $\geq 2$ semantically orthogonal
verifiers with pre-emptive rotation:

1. **$E_1$:** Z3 satisfaction / type checks (safety / soundness)
2. **$E_2$:** liveness property checks OR semantic AST density / structural
   complexity (intent / non-triviality)
3. (Optional) **$E_3, ..., E_k$:** spec-grounded behavioural checks,
   property-based test results, etc.

The rotator picks $i^* = \arg\max_i \widehat\Phi_i$ at each round.

### Decentralization-respecting design rule update

CLAUDE.md's argument that "constraint-derived energies are structurally
protected" needs nuance. Boolean Z3 energies are protected against
*continuous* gaming but vulnerable to *specification gaming on the
null space*. The full structural protection is: **Boolean
constraint-derived energy + multi-verifier rotation**. Either alone
is insufficient.

### Phase 2 hardware mandate (refined)

Phase 2 hardware acceleration enables:
- Reaching $\delta_{\text{plateau}}$ efficiently for any single $E_i$
- Maintaining the rotation cadence at the per-round Sonnet+Opus budget
  required by FR-11 self-distillation

The Q1 closed-form gives concrete sizing: each $E_i$ has its own
plateau at $\delta_0 \|\Pi_{E^\perp_i} g_0\| / \sqrt{1 - (\varepsilon/b_{1,i})^2}$.
Plateau is defeated *between rounds* via rotation, not by within-round
sampling.

### Publishable results

Three publishable contributions emerge:

1. **The orthogonality stall closed-form** (Q1) with the explicit
   $\varepsilon/b_1$ correction term. Likely a novel theorem.
2. **Ensemble vs rotation:** ensemble combine creates frustrated
   landscapes; rotation strictly dominates. Counter-intuitive
   engineering result.
3. **The null-space mimicry attack and rotation defence:** statistical
   tests on a single verifier are insufficient; multi-verifier pre-emptive
   rotation is the only safe architecture. This is a security-relevant
   foundation-model architecture finding.

## Action plan

### Immediate updates

1. **Change proposal `zenil-grounded-self-distillation-deployable-stack.md`:**
   - Update Exp E: rotation architecture (NOT ensemble), pre-emptive
     cadence (NOT on-stall).
   - Add Exp E.2: null-space mimicry attack + rotation-defence
     demonstration on synthetic adversarial agent.
   - Update REQ-PHASE3-001 spec: "Phase 3 self-distillation MUST
     include $\geq 2$ semantically orthogonal verifiers with pre-emptive
     rotation as the only defence against null-space mimicry."

2. **Memory updates:**
   - Update `project_orthogonality_stall.md` with the closed-form
     plateau and the rotation requirement.
   - New memory entry: `project_null_space_mimicry_attack.md`
     capturing the attack + defence.

3. **`_bmad/architecture.md` "Asymptotic Hardware Mandate":**
   - Add the explicit δ_plateau formula.
   - Note that ensemble combine is *not* a viable Phase 3 architecture
     (frustrated landscape blowup).
   - Pre-emptive rotation is the architectural mandate.

4. **CLAUDE.md decentralization rules:**
   - Update Rule 7 (no vendor abstractions) with the nuance: Boolean
     constraint-derived $E$ + multi-verifier rotation is the full
     protection. Either alone is insufficient.

### Open questions for future Deep Think rounds

- **Q4:** what's the *minimum semantic orthogonality* required between
  $E_1$ and $E_2$ for rotation defence to work? Quantifiable as a
  Friedrichs angle bound between their kernel-projections?
- **Q5:** does the rotation defence have any blind spots? Specifically,
  can an adversary construct trivial outputs that simultaneously land
  in $\bigcap_i \mathrm{null}(E_i)$? If yes, what's the minimum $k$
  for which this becomes computationally infeasible?
- **Q6:** how does the orthogonality stall interact with the Zenil
  α_t lower bound? Specifically, does $\inf_t \alpha_t > 0$ hold under
  multi-verifier rotation, or only under stronger conditions?

## What we got right

- Q1 leading dependence (proportional to orthogonal mass)
- Q3 internal-statistics approach (different formulation but valid direction)
- Q3(c) Z3 protection (right intuition; needed nuance)

## What we got wrong

- Q2(b) rotation cadence — wrong direction (we said on-stall, correct is pre-emptive)
- Q2(c) ensemble vs rotation — wrong (we said ensemble dominates if uncorrelated)
- Q2(a) sufficiency of null-space coverage — wrong (need quantitative angle)

## What surprised us

- **Ensemble combine creates frustrated MCMC landscapes** — non-obvious.
- **Boolean $E$ neutralises the variance-suppression test** — Z3's protection
  against continuous gaming creates a vulnerability to null-space mimicry
  that wasn't in our threat model.
- **The Q3 test has zero power against the worst-case attack** — we predicted
  the test would discriminate; in fact it's blind to specification gaming.

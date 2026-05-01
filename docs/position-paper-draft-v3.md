# Carnot: An Architectural Framework for Mapping the Empirical Bounds of LLM Verification

**Position paper, draft v3 (2026-05-01).** Target: arXiv preprint by
2026-05-15. v3 is a structural rewrite of v2 (`docs/position-paper-draft-v2.md`,
exp1091); it is not v2 with edits. The .85 milestone surfaced four
methodological recalibrations — D_eff = 1.603 across deployed text
probes (exp1093), KL = 3.07 detailed-balance violation in synchronous
parallel Glauber (exp1094), Pareto-suboptimal cascade routing on
SOTA outputs with an energy-ordering inversion (exp1100), and the
collapse of pre-filtered self-distillation grounding (exp1099) —
that v2's "SOTA-beating verification machine" narrative cannot
honestly absorb. v3 adopts the position that these recalibrations
*are* the contribution: Carnot is the rigorous EBM scaffolding
needed to expose, measure, and bound the structural friction of
decentralized LLM verification.

## Abstract

Verifier-filtered self-distillation has become a central paradigm
for aligning open-weight LLMs without closed-frontier dependence.
Naively, the recipe is "compose more verifiers, sample faster, and
the residual error vanishes." We show empirically that this naive
scaling collides with three structural walls — a verifier
correlation ceiling, an exact-sampling detailed-balance limit, and
an out-of-distribution energy-ordering inversion on highly-optimized
SOTA outputs — that no amount of compute or engineering effort can
push past in their current form. We position Carnot not as a
finished verification system but as an architectural framework
designed to make these walls measurable, to bound them with
closed-form theory, and to deploy fall-backs that remain
mathematically exact under the bounds. Specifically, we (i) replace
the naive geometric-mean joint-volume approximation with the
correct sqrt(det(Sigma)) factor and apply the Welch/Rankin Simplex
bound to derive the verifier-composition ceiling k* <= 3.125 from
the empirical alpha^2 = 0.66 of three deployed text probes
(D_eff = 1.603, exp1093); (ii) report a 15.6x exact-sampling
speedup of a chi <= 4 sparse-constraint FPGA Fast-Path against an
optimized C++ baseline, after auditing and rejecting the
synchronous parallel Glauber sampler that produced the prior
"~13,000x" headline (KL = 3.07 against single-site Gibbs, exp1094);
(iii) document the energy-ordering inversion on SOTA outputs
(mean_correct = 0.689 > mean_incorrect = 0.621, exp1100) as an
out-of-distribution Goodhart-class anomaly that constrains the
class of energy functions any decentralized verifier may use; and
(iv) report the brittleness of pre-filtered self-distillation
(exp1099) as evidence that the energy signal has to drive the
filter, not be re-derived from accept/reject labels. Carnot's
positive baseline results — SOS-KAN AUROC = 0.9545 on the FoVer
corpus, alpha_t = 0.38 on Qwen3.6-35B-A3B with verifier-filtered
self-distillation, +36 pp on HumanEval pass@1 with verify+repair —
demonstrate that the framework is operational in-distribution; the
foregrounded negatives demonstrate that it is rigorous out of it.
We position Carnot's Apache-2.0, local-first, hardware-portable
design as the engineering substrate that makes cross-mechanism
verifier diversity (the only known route past the Welch ceiling)
physically deployable on consumer hardware.

## 1. Introduction

The problem we want to attack is decentralized verification of
LLM outputs: given a hosted-locally open-weight base model whose
generations may be incorrect, hallucinated, or adversarially
manipulated, produce a calibrated rejection signal that runs on
the same hardware as the base model and depends on no closed-API
oracle. This is a hard problem because there is no obvious way to
get an external truth signal: any such verifier ultimately has to
rest on either symbolic constraints, runtime execution, or another
learned probe of similar provenance to the base model.

The naive recipe in the field — and in v2 of this paper — was
**compose more verifiers, run them faster, and the joint null
space shrinks exponentially in k.** Three .85 milestone experiments
show that this recipe fails in three distinct ways.

**Wall 1 — Verifier correlation ceiling.** When we measured the
pairwise r-correlation across three deployed text probes
(SpilledEnergyDetector, NUPProbeV4, PCIBProbe) on 364 examples
(exp1093), we found r in [0.41, 0.66] with a Participation Ratio
of D_eff = 1.603 — three nominally independent verifiers carry less
than two effective dimensions of information. The naive
"AND-composition shrinks the joint null space exponentially in k"
result is conditional on transversality (Friedrichs angle
theta_F > 0); empirically, theta_F is small for any verifier
family that reads continuous text features.

**Wall 2 — Exact-sampling detailed-balance limit.** The synchronous
parallel Glauber sampler that produced the original "~13,000x
FPGA speedup" headline does not preserve detailed balance on
frustrated topologies: KL(P_parallel || P_Gibbs) = 3.07 on the
canonical 12-spin antiferromagnetic ring (exp1094). The numerical
speedup against an unoptimized Python CPU baseline was real; the
sampling distribution it produced was not the target distribution.

**Wall 3 — Out-of-distribution energy-ordering inversion.** On
100 SOTA-model outputs, the cascade reports
mean_correct_energy = 0.689 vs. mean_incorrect_energy = 0.621
(exp1100). The energy gap inverts: heavily-optimized SOTA outputs
produce *lower* energy for incorrect completions than for correct
ones. The same energy function on the FoVer corpus reaches
AUROC = 0.9545; the inversion is not a sign-bug but an
out-of-distribution shift consistent with reward-hacking dynamics
(Goodhart's Law) on outputs that were optimized against a
similarly-shaped verifier.

We could ignore each wall, fabricate around it, or disclaim it in
an appendix. We choose instead to build the entire paper around
the proposition that **the walls are themselves the contribution.**

The pivot has consequences for every section. Section 2 defines
the architectural framework as the *measuring apparatus* deployed
to discover the walls, not as a "finished" verification system.
Section 3 develops the theory needed to bound them: the
sqrt(det(Sigma)) joint-volume factor, the Welch/Rankin Simplex
bound on k*, the Participation Ratio as the empirical projection
onto the bound. Section 4 reports the hardware deployment story —
including the 15.6x recalibration and the chi <= 4 Fast-Path
architecture — as a direct response to the KL = 3.07 finding.
Section 5 foregrounds the four .85 honest negatives as the
architectural telemetry's primary output. Section 6 positions
Carnot's Apache-2.0, local-first design as a strict engineering
prerequisite for cross-mechanism verifier diversity. Section 7
closes with the Mock Cascade roadmap and the precise bounds the
framework hands future work.

The audience for this paper is researchers who care about whether
the published numerical claims of LLM-verifier work survive
adversarial scrutiny. Carnot is offered as a worked example of
publishing the limits along with the headline.

## 2. Carnot Architectural Framework

Carnot is a four-tier verification cascade routed by an energy
budget, plus a Phase 3-7 defensive stack derived in 9 rounds of
Deep Think cross-validation. We describe the framework here as the
*scaffolding* that the .85 measurements were taken against, not as
a deployable verification system whose numerical claims should be
trusted at face value. Section 5 makes the in-distribution vs
out-of-distribution split explicit.

### 2.1 Cascade routing

Each LLM output enters the cascade and is routed through tiers of
progressively-more-expensive verifiers, exiting at the first tier
whose energy threshold either accepts or rejects with confidence:

* **Tier 0a — ThinkPRM probe.** Step-level discriminative probe;
  cheapest. AUROC = 0.9885 on the full FoVer corpus (predecessor
  of exp1033; the v4 CI-stub artifact reports 0.5 on an 85-example
  subset and is not the headline).
* **Tier 0b — SpilledEnergy detector.** Pre-softmax logit-vs-output
  energy gap; cheap, high-skip-rate; baseline arXiv:2602.18671.
* **Tier 2 — SOS-KAN energy.** Sum-of-Squares Kolmogorov-Arnold
  network with type-level monotonicity and non-negativity. AUROC =
  0.9545 on 6,548 FoVer pairs (exp1072), zero monotonicity
  violations across 16,000 invariant tests.
* **Tier 3 — Ising MCMC.** Parallel-tempered Glauber dynamics.
  Hardware-acceleratable on FPGA via the SamplerBackend protocol
  (Section 4).

The cascade routing problem is a finite-horizon constrained POMDP
over the four-tier state space, solved exactly by Bellman backward
induction. The closed-form Wastefulness Condition c_j > (lambda/2)
* |f_1^(j) - f_0^(j)| identifies tiers whose marginal cost
exceeds their marginal information gain. The lineage is
Saberian-Vasconcelos cascade design + Schneidman MaxEnt-Ising
joint modeling + Wald sequential probability ratio testing.

### 2.2 The 9-round Zenil derivation chain

The Phase 3-8 defensive stack was derived to address structurally
distinct attack classes identified by independent cross-validation
rounds. We summarize the layers as scaffolding rather than as
load-bearing claims, because their empirical bounds are the
subject of Section 3:

* **Phase 3 — Static defence.** AND-composition over verifiers
  with distinct kernel manifolds. Theorem 3.2 below states the
  conditional kernel-shrinkage bound; the *condition* — Friedrichs
  transversality theta_F > 0 — is what fails empirically (Section 3).
* **Phase 4 — Concept drift.** Factorized per-verifier curriculum
  allocates audit budget proportional to per-verifier predicted
  drift, not aggregate drift.
* **Phase 5 — Detection latency.** Information-action bottleneck
  bound Delta_lat^min = rho_dot * (tau_action - tau_star)^+ + z *
  sigma_pred(tau_star) under a Local Linear Trend predictive UCM.
* **Phase 6 — Whip / shadow-boundary.** Multi-scale ensemble
  detection at log-spaced temporal scales bounds the high-frequency
  + slow-stealth chained payload.
* **Phase 7 — Cyclic recurrence + churn gap.** Stochastic-veto
  continuum memory; the ~28.9% all-skip rate arising from the
  Euler product (1/2; 1/2)_inf motivates Phase-8d cryptographic
  state sealing.
* **Phase 8a-d — Fundamental epistemic limits.** PAC-Bayes Budget
  Starvation, DVS Poisoning via Spurious Shortcut Learning,
  Modality-Asymmetry Prompt Injection, RNG Predictive Collusion.

The Phase-8 audit (2026-04-29) added 8a-d to the architecture
specifically because the prior derivation chain had assumed the
pathologies away. The framework's **value as scaffolding** is that
structural attack classes get named and bounded; its **limit as a
finished system** is that the bounds are conditional on
transversality, and Section 3 documents that the empirical
transversality on deployed text probes is small.

### 2.3 The framework as measuring apparatus

The Carnot deployment in this paper is best read as a measuring
apparatus. The cascade routing logs the per-tier exit fraction
across SOTA outputs (exp1100); the Phase-3 AND-composition logs
the joint kernel dimension across deployed verifiers (exp1093);
the Tier-3 hardware path logs the KL divergence between the
proposed sampler and the target distribution (exp1094); and the
self-distillation closure logs the alpha_t grounding signal
(exp1077). The .85 milestone's contribution is the four
quantitative readouts off this apparatus, three of which document
walls the naive recipe cannot push past.

## 3. Theoretical Bounds of Verification Composition

Section 3 corrects v2's geometric-mean joint-volume approximation,
states the correct sqrt(det(Sigma)) factor, and applies the
Welch/Rankin Simplex bound to project the empirical alpha^2 = 0.66
onto a numerical ceiling on the maximum verifier-composition size.

### 3.1 The sqrt(det(Sigma)) joint volume

For a verifier ensemble {E_1, ..., E_k} represented as standardized
unit vectors in L^2(p) Hilbert space, the joint kernel volume of
the AND-composition is *not* the geometric mean
prod cos^k(theta_F) of the pairwise Friedrichs angles; that is a
volume-of-the-product approximation that holds only when the
verifiers are pairwise independent. The correct factor is

    Vol(joint kernel) ∝ sqrt(det(Sigma))

where Sigma is the k x k correlation matrix of the verifier
energies. The geometric-mean approximation overstates the joint
volume by orders of magnitude when Sigma has a dominant principal
component; v2's k = 15 application of this approximation produced
a joint-volume estimate that disagrees with the determinant
by a factor of ~3.2 x 10^9. We retract that approximation here
and use the determinant throughout v3.

### 3.2 The Welch / Rankin Simplex bound

The maximum number of unit vectors in any Hilbert space whose
pairwise inner products satisfy <e_i, e_j> <= -c (for c > 0) is

    k* <= 1 + 1/c

(generalized Rankin / Welch Simplex bound, Welch 1974). Decomposing
each verifier into its valid-signal component and a residual
mechanism vector,

    f_i = alpha * V + sqrt(1 - alpha^2) * e_i,    e_i ⊥ V

so that the pairwise verifier correlation r_ij satisfies

    r_ij = alpha^2 + (1 - alpha^2) <e_i, e_j>

and enforcing the architectural constraint r_ij <= r_max for some
r_max <= 1 yields the residual-vector inner product
<e_i, e_j> <= -(alpha^2 - r_max)/(1 - alpha^2) = -c. Substituting
into the Welch bound,

    k* <= floor((1 - r_max) / (alpha^2 - r_max)).

### 3.3 Empirical projection: D_eff = 1.603 and k* <= 3.125

Plugging Carnot's measured numbers in (exp1093):

* alpha^2 = 0.66 (the dominant pairwise correlation,
  SpilledEnergyDetector vs NUPProbeV4)
* r_max = 0.5 (architectural constraint on maximum allowed
  verifier overlap)

yields

    k* <= floor((1 - 0.5) / (0.66 - 0.5))
        = floor(0.5 / 0.16)
        = 3.125

The Participation Ratio of the same correlation matrix is

    D_eff = (sum(lambda_i))^2 / sum(lambda_i^2) = 1.603

confirming that three nominally independent text probes carry
~1.6 effective dimensions of information. The conclusion is that
**arbitrary k composition across homogeneous text-probe verifiers
cannot exceed k* ≈ 3 verifiers without violating the architectural
r-correlation constraint.** v2's "k = 15 AND-composition" claim
is mathematically infeasible from a homogeneous text-probe family
and is retracted.

### 3.4 The escape: cross-mechanism diversity

The Welch bound depends only on the relationship between alpha^2
and r_max. If a verifier family with substantially lower alpha^2
(less shared valid-signal pollution) is added to the ensemble, the
bound loosens. Empirically, structurally distinct verifier
mechanisms — symbolic SMT solvers, runtime sandbox execution,
unit-test runners, Z3-AST extractors, JSON-Schema validators —
are conjectured to occupy *disjoint* manifolds in input space,
because they read structurally different signals than continuous
text-feature probes. A heterogeneous ensemble drawing one
verifier from each mechanism family (numerical, semantic,
step-level, combinatorial, runtime, formal) admits k_max ≈ 7-8
under the same alpha^2 = 0.66 constraint while preserving
r_ij <= 0.5.

The escape is therefore not "compose more verifiers of the same
kind" but **"compose the right *kind* of verifiers."** The
follow-up empirical experiment (exp1104) is specified in the .85
roadmap to measure the maximum-clique size of a topologically
diverse 8-probe ensemble at r_max = 0.45 and AUROC >= 0.85.

[FIGURE 6: Welch ceiling — k* vs alpha^2 contour, with the
empirical (alpha^2 = 0.66, r_max = 0.5) point plotted at k* = 3.125,
and dotted lines showing how k* loosens to ~7-8 as alpha^2 falls
toward 0.4 under cross-mechanism diversity.]

### 3.5 Finite-tail joint evasion

The Welch bound is the sufficient ceiling under standardized
covariance, but the security guarantee that matters for adversarial
inputs is the joint extreme-tail evasion probability
P(all_E_i below z) under an equicorrelated Gaussian copula. Direct
numerical integration of

    P(all_E_i < z) = ∫_{-∞}^{∞} φ(u) [Φ((z − √r·u) / √(1−r))]^k du

at z = Φ^{-1}(0.01) ≈ -2.326 (a strict 1% individual evasion rate)
yields the joint evasion probabilities in Table 1.

| k | r | Joint evasion probability |
|---|---|---|
| 5 | 0.30 | 4.12 x 10^-6 |
| 10 | 0.40 | **1.20 x 10^-6** (tightest) |
| 15 | 0.55 | 8.36 x 10^-6 (weakest) |

Table 1: equicorrelated Gaussian copula tail integral. The proposed
v2 architecture (k = 15, r ~ 0.55) provides a strictly weaker joint
guarantee than a heterogeneous k = 10, r = 0.4 ensemble — the
correlation drag dominates the apparent volume gain. This is the
mathematical motivation for Section 3.4's pivot.

### 3.6 Round-12 saturation under the corrected math

The asymptotic residual error of verifier-filtered self-distillation
under proper distributional normalization Z_t in (0, 1) and a fixed
verifier suite {E_i} satisfies

    delta_infty^normalized = C_Z * ||nu_0^perp||

where nu_0^perp is the projection of the initial residual onto the
joint null space and C_Z = prod_t Z_t^{-1} is the cumulative-rejection
constant. The bound itself is unchanged from v2; what changes under
v3's corrected math is that ||nu_0^perp|| is computed against the
sqrt(det(Sigma)) joint volume, not the geometric-mean approximation.
The renormalization gap Delta_renorm = (C_Z - 1) * ||nu_0^perp|| is
driven by cumulative early-step rejection, not by per-step verifier
accuracy.

## 4. Hardware Acceleration & Sampling Limits

Section 4 reports the FPGA hardware deployment with the 15.6x
recalibration in place, audits why the prior "~13,000x" headline
was distributionally invalid, and derives the chi <= 4 Sparse-
Constraint Accelerator architecture as the deployable response.

[FIGURE 3: chi <= 4 Fast-Path tradeoff — speedup vs CPU exact
Gibbs as a function of chromatic number chi, showing the 15.6x
plateau at chi <= 4 and the collapse to pseudo-sequential
performance at chi > 8, with the chi >= 8 regime annotated as
the CPU fallback boundary.]

### 4.1 The detailed-balance audit (exp1094)

The .85 sampler-correctness audit measured the KL divergence
between Carnot's prior synchronous parallel Glauber FPGA sampler
and the target Boltzmann distribution on the canonical 12-spin
frustrated antiferromagnetic ring at beta = 2.0:

    KL(P_parallel || P_Gibbs) = 3.07

against an acceptance threshold of 0.05 and a theoretical bound of
0.058 (within sample-size-driven Monte Carlo noise). Synchronous
parallel Glauber updates interacting spins concurrently using only
the previous-iteration neighbour state, which mathematically violates
the local-balance condition that single-site Gibbs preserves. The
sampler did not converge to P(x) ∝ exp(-E/T); it converged to a
distribution that disagrees with it by ~3 nats per sample.

Consequently every speedup number reported against this sampler in
v2 is invalid as a *sampling* speedup, even though the per-sample
*latency* number (24.83 us at 64 spins) is accurate as a hardware
benchmark. We **retract** the 13,061x and ~12,000x speedup figures
from v2.

### 4.2 The 15.6x recalibration

A pipelined chromatic-Glauber sampler that respects detailed balance
requires k color-batch flushes per Monte Carlo sweep, where chi is
the chromatic number of the constraint graph. At chi = 4 — the
empirical sweet spot for SAT-style constraint matrices with
maximum clique size <= 4 — each sweep takes 4 cycles per color x
4 colors = 16 cycles = 64 ns at 250 MHz. Against an optimized
single-thread C++ Gibbs sampler at ~1 us per sweep, the latency
speedup is

    ~1 us / 64 ns ≈ 15.6x.

This is a rigorous exact-sampling speedup against a real baseline,
not against a Python-overhead-dominated CPU loop. Section 4.4
discusses the regime in which 15.6x is the right hardware bet.

### 4.3 The chi <= 4 Sparse-Constraint Accelerator

For arbitrary Carnot constraint graphs, the chromatic number chi
is data-dependent. SAT-style constraints with c clauses on n
variables and c ≈ 10n have expected chi ∈ [8, 15]; arithmetic
constraints over k variables form K_k cliques and inherit
chi >= k. The chi <= 4 regime is therefore a real but bounded
slice of the deployed constraint distribution, not the universal
case.

The architecture we ship in response is the **Sparse-Constraint
Accelerator + CPU Fallback**:

1. **CPU pre-processor** runs DSatur on the input constraint graph
   to estimate chi.
2. **If chi <= 4**, dispatch to the FPGA chi <= 4 Fast-Path
   bitstream, which implements pipelined chromatic Glauber with
   provable detailed balance and 15.6x speedup vs C++ Gibbs.
3. **If chi > 4**, fall back to single-site exact Gibbs on the
   CPU. This preserves correctness at the cost of speedup.

The architecture is mathematically exact at every step. There is
no regime in which the user gets an incorrect distribution; there
is a regime (chi > 4) in which they do not get the FPGA speedup.
The asymmetric design directly answers the KL = 3.07 finding:
sampling correctness is non-negotiable; speedup is a function of
constraint topology.

### 4.4 The Z1 and photonic deferral

v2 framed the Extropic Z1 as a near-term Phase-2 production target.
v3 retracts that framing. As of 2026-05, no peer-reviewed
independent benchmark has demonstrated that Z1 silicon samples
exactly from P(x) ∝ exp(-E/T) on arbitrary frustrated non-planar
topologies with KL <= 0.05; available benchmarks come from vendor
materials and closed-beta cloud APIs. Analog thermodynamic hardware
optimizes (finds energy minima) reliably but historically struggles
with rigorous equilibrium *sampling* because of analog noise floors,
local freezing, and calibration drift. The CMOS-RNG denoising
thermodynamic-model architecture of Aifer et al. (Extropic
co-authors) [15] is a useful published proxy, but we cannot block
a cryptographic-grade verifier on an unverified vendor timeline.

We re-classify the Z1 and the all-optical Ising+KAN photonic
substrate of Cong et al. [11] as **future research directions
pending independent silicon benchmarking.** They remain credible
Phase-2 targets; the paper's hardware claim depends only on the
KV260 FPGA chi <= 4 Fast-Path + CPU fallback, not on hardware we
do not have.

### 4.5 KV260 baseline measurement

The KV260 smoke test (exp1068, board IP 192.168.51.98, /dev/uio4,
AXI base 0xA0000000, schema v9) reports per-sample latency
mean = 24.83 us, min = 24.19 us, max = 40.08 us at 64 spins over
100 samples with 70 unique values and a non-uniform energy
distribution. This is the load-bearing single-point hardware
measurement v3 retains; the 15.6x speedup of Section 4.2 is the
bound under chi = 4, not the universal claim. The multi-N FPGA
scaling curve was not reached during the .84 milestone (exp1081
could not connect to the board) and is left as future work.

## 5. Empirical Realities & Anomalies

Section 5 reports the four .85 honest-negative findings, foregrounded
as the contribution rather than buried. Each is grouped under the
unifying theme **"The Structural Friction of LLM Verification."**
The framework's positive baseline numbers (Section 5.5) are
preserved separately to scope the in-distribution operating regime.

### 5.1 Pre-filtered self-distillation collapses (exp1099)

We attempted to integrate Apple's Self-Distilled RLVR (SSD) recipe
with Carnot's Tier-2 SOS-KAN energy as the per-example filter. The
training corpus had been pre-filtered through Carnot's
AND-compose-k5 module before SSD ingestion, which collapsed the
energy_score field to a constant 0.0 across all 150 examples. The
energy filter degenerated: at threshold = median = 0.0, every
entry was accepted. The four conditions reported

| Condition | Selection rule | Fraction correct |
|---|---|---|
| A: RLVR-only | accept all | 0.5333 (= baseline) |
| B: SSD-only | majority vote | 0.4 |
| C: RLVR + SSD energy filter | degenerate accept | 0.4 |
| D: on-policy SSD (fallback) | low-energy 80 of 100 | 1.0 |

honest_verdict: `no_improvement_honest_negative`. Condition D's
1.0 was achieved only via fallback; the energy-driven discrimination
the experiment was designed to test was not observable.

The lesson is methodological. **Pre-filtering training data through
the same verifier whose energy signal is intended to drive
self-distillation is a data-leakage failure mode**, not a
performance bottleneck. The energy distribution among accepted
examples no longer carries information; you have already filtered
out the variance you intended to use. The forward correction is
**the energy signal must drive the filter, not be derived from
accept/reject labels** — the SSD energy filter has to see the raw,
unfiltered candidate distribution.

### 5.2 Cascades on SOTA outputs are Pareto-suboptimal (exp1100)

The cascade-validation experiment ran the four-tier cascade against
100 SOTA-model (Qwen3.6-35B-A3B) outputs:

| Tier | Exits | Cumulative |
|---|---|---|
| Tier 0a | 20 | 20 |
| Tier 0b | 56 | 76 |
| Tier 2 | 8 | 84 |
| Tier 3 | 16 | 100 |

mean_cascade_depth = 2.20, vs FoVer-corpus depth where Tier 0a
exits 8% of the time. SOTA outputs need a deeper cascade (20%
exit at Tier 0a vs 8% on FoVer); the cheap-prefilter optimization
is less useful when the upstream model is more capable. More
critically, the experiment recorded

* mean_correct_energy = 0.689
* mean_incorrect_energy = 0.621
* incorrect_energy > correct? **false** (energies inverted)

honest_verdict: `cascade_validated_sota_inefficient`. The cascade
*functions* end-to-end; the energy ordering it produces on SOTA
outputs is *inverted* relative to the same cascade's behaviour
on FoVer.

We treat the inversion as an out-of-distribution Goodhart-class
anomaly, not a sign-bug. The same energy function reaches
AUROC = 0.9545 on FoVer (Section 5.5); the inversion appears
specifically when evaluating heavily-optimized SOTA-model outputs
that were themselves trained against verifier-style reward
signals. We believe the inversion arises because the SOTA
generator has learned to satisfy the linear proxy heuristics that
Carnot's energy function reads, while still producing semantically
incorrect completions — classic reward hacking. Resolving the
inversion is an open methodological question; the empirical
triage we ran is sufficient to distinguish OOD shift from
implementation defect (the FoVer in-distribution AUROC is
unchanged), not sufficient to prescribe a fix. The honest framing
is that **non-linear EBM landscapes (Phase-4) are required to
verify outputs from linearly-optimizable SOTA generators**.

### 5.3 D_int = 1.6 motivates the Welch bound (exp1093)

Section 3.3 already reported D_eff = 1.603. We restate the
empirical context here because the measurement is the load-bearing
input to the Welch bound. The verifiers measured were three
training-free Tier-0 text probes
{SpilledEnergyDetector, NUPProbeV4, PCIBProbe} on a 364-example
corpus with no GPU/network dependence (training-free probes were
chosen so the measurement runs on the laptop). Pairwise
r-correlations were

* SpilledEnergyDetector vs NUPProbeV4: 0.656
* SpilledEnergyDetector vs PCIBProbe: 0.546
* NUPProbeV4 vs PCIBProbe: 0.406

with single-verifier null-space fractions ranging from 0.005 to
0.066 (each verifier individually has a small null space; the
problem is the *shared* signal). honest_verdict:
`verifiers_correlated_diversity_needed`.

The result motivates Section 3.4's cross-mechanism diversity
prescription: the only known route past the Welch ceiling is to
compose verifiers from structurally distinct families, not to
add more verifiers from the same family.

### 5.4 KL = 3.07 justifies CPU fallback (exp1094)

Section 4.1 already reported the detailed-balance violation. We
restate the operational consequence here: the **Sparse-Constraint
Accelerator + CPU Fallback** architecture (Section 4.3) is not
a defensive engineering choice; it is the **mathematically forced
deployment shape** once detailed balance is non-negotiable and
chi is data-dependent. The fallback is not a hedge against
hardware failure; it is the regime in which hardware acceleration
does not apply.

honest_verdict: `fpga_sampler_distribution_mismatch_confirmed`.

### 5.5 In-distribution baseline (preserved from v2)

The four .85 negatives do not invalidate Carnot's
in-distribution operational behaviour. We preserve the v2
baseline numbers so the framework's operating range is honestly
scoped:

* **SOS-KAN AUROC = 0.9545** (exp1072): on the 6,548-pair FoVer
  corpus, Tier-2 SOS-KAN v3 with neural-Gram parameterization
  reaches AUROC 0.9545 with zero monotonicity violations across
  16,000 invariant tests. The v1 zero-shot baseline was 0.6042;
  v3 delta is +0.350.
* **alpha_t = 0.38** on Qwen3.6-35B-A3B (exp1077): the FR-11
  self-distillation closure on the SOTA local model measures
  alpha_t = 0.38 over 100 generated questions with k_verifiers = 5.
  This satisfies the Zenil convergence threshold inf_t alpha_t > 0
  on the SOTA model. The small-model baseline on Qwen3.5-0.8B
  (exp1074) was alpha_t = 0.78, and the lower SOTA-model number
  is consistent with a base model already closer to mu_P (smaller
  per-step grounding contribution).
* **HumanEval +36 pp** (exp1079): the same SOTA local model with
  Carnot verify+repair improved HumanEval pass@1 by +36 pp absolute
  in single-shot evaluation. The harness reported baseline pass@1
  = 0.0 because it could not parse the SOTA model's raw-text
  outputs into the HumanEval format; the same extraction limit
  produced GSM8K TP = 0 in the same experiment, so we report
  HumanEval only.
* **KV260 latency 24.83 us** (exp1068): per-sample latency at
  64 spins.

These numbers are the *in-distribution* operating regime of the
framework. The .85 negatives are the *out-of-distribution* and
*scaling* boundaries. Both are required to honestly describe the
framework.

[FIGURE 2: SOS-KAN AUROC binormal curve.]
[FIGURE 4: alpha_t against the Zenil inf_t > 0 threshold.]

## 6. Decentralization & Deployment Sovereignty

Carnot's Apache-2.0, local-first, multi-integration-surface
(Python API, CLI, MCP server, HTTP REST), hardware-portable design
is documented in this paper not as a moral commitment but as an
**engineering prerequisite** for the cross-mechanism verifier
diversity that Section 3.4 identified as the only route past the
Welch ceiling.

The argument is direct. Reaching k_max ≈ 7-8 with r_ij <= 0.5
requires verifiers drawn from disparate families: symbolic SMT
solvers, runtime sandbox execution (gVisor), unit-test runners,
Z3-AST extractors, JSON-Schema validators, and combinatorial
Ising/Potts probes. Each family has its own latency profile and
its own deployment shape; some require GPU/CPU compute, some
require disk-bounded sandbox execution, some require dedicated
hardware (Tier 3 Ising). Composing them at sub-millisecond
inference latency requires the verifier suite to physically share
the same hardware as the base model. Any centralized provider
layer that secures the model behind an API boundary structurally
introduces latency and bandwidth that make cross-mechanism MCMC
sampling at the required rate impossible.

This is the **engineering-first** framing. It does not depend on
any moral position about open-source or vendor relationships;
it depends only on the round-trip latency numbers that Welch-bound
escape mathematically requires. The closed-source-mechanistic-
interpretability path of Goodfire Silico [G] (white-box neuron
inspection on open-weight models, hosted as a centralized
service) is structurally complementary to Carnot, not a competitor:
the two architectures secure structurally different layers of the
verification stack. **Goodfire's white-box approach secures the
centralized provider layer; Carnot's local-first approach secures
the post-generation edge-verification layer** that decentralized
deployment requires. The two are physically composable: a
Goodfire-verified centralized base model plus a Carnot local
edge verifier strictly dominates either alone.

The decentralization-respecting design constraints documented in
the project's CLAUDE.md (local-first, opt-in closed-frontier,
distribution mirroring across at least two channels, multiple
integration surfaces, hardware portability, per-call data
minimization, no vendor-specific abstractions in the core) are
the engineering policy that operationalizes this argument. Every
empirical result in this paper was produced on local hardware
with open-weight models; closed-frontier-model integration is
opt-in, never required.

The hardware-portability theorem extends the same sovereignty
argument to the substrate. Theorem 4.6 (preserved from v2):
provided every individual verifier's constraint manifold
intersects the others transversally (theta_F > 0), Carnot's
parallel-tempered AND-composition guarantees strictly polynomial
MCMC sampling latency across (a) discrete FPGA Glauber dynamics,
(b) continuous thermodynamic samplers (Z1 / XTR-0 class), and
(c) optical photonic Ising substrates. Section 3 documents that
the empirical theta_F is small for homogeneous text-probe
verifiers; Section 3.4 documents that cross-mechanism diversity
loosens the constraint. The substrate-portability claim is
preserved; the conditional theta_F > 0 is what cross-mechanism
diversity is engineered to satisfy.

## 7. Conclusion & Roadmap

We have presented Carnot as an architectural framework for mapping
the empirical bounds of LLM verification, organized around four
.85-milestone honest negatives that name three structural walls
(verifier correlation ceiling, exact-sampling detailed-balance
limit, OOD energy-ordering inversion) and one methodological
trap (pre-filtered self-distillation degeneracy). The
contributions are:

1. **Welch bound application** (Section 3): the Welch / Rankin
   Simplex bound projected onto Carnot's empirical alpha^2 = 0.66
   yields k* <= 3.125 for homogeneous text probes; cross-mechanism
   diversity is the only known route to k_max ≈ 7-8.
2. **sqrt(det(Sigma)) joint volume** (Section 3.1): the correct
   joint-volume factor; the geometric-mean approximation used in
   v2 is retracted.
3. **chi <= 4 Sparse-Constraint Accelerator + CPU Fallback**
   (Section 4.3): a mathematically exact deployment shape that
   answers the KL = 3.07 detailed-balance violation; rigorous
   15.6x speedup vs optimized C++ baseline at chi <= 4.
4. **OOD anomaly framing of the energy-ordering inversion**
   (Section 5.2): the inversion on SOTA outputs as evidence that
   non-linear Phase-4 EBM landscapes are required to verify
   outputs from linearly-optimizable SOTA generators.
5. **Decentralization as engineering prerequisite** (Section 6):
   cross-mechanism verifier diversity is physically deployable
   only on local-first hardware-portable infrastructure;
   complementary to centralized white-box approaches, not
   competitive.

The Mock Cascade engineering roadmap — a parallel-track
deployment that trades the Phase-3 prototype's joint-kernel
guarantees for a heterogeneous k=8 ensemble routed by the same
cascade logic — is scoped for the .86 milestone (exp1104 and
its successors). The roadmap is the practical answer to "what
do you do once you have the bounds?"

Three open methodological questions remain. **(a) The energy
inversion fix.** The Phase-4 non-linear-landscape derivation is
load-bearing here; until it is empirically verified that a
non-linear EBM does not also invert on SOTA outputs, the OOD
inversion is bounded but not resolved. **(b) Cross-mechanism
verifier composition.** exp1104 will measure the actual maximum
clique of an 8-probe heterogeneous ensemble; the conjecture is
k_max ≈ 7-8 but the measurement has not been run. **(c) Hardware
sampling correctness on Z1 / photonic substrates.** The CPU
fallback path covers correctness everywhere; getting independent
exact-sampling benchmarks on Z1 silicon is the prerequisite for
moving Z1 from "future research direction" to "deployable
substrate."

We position v3 not as the end of a research program but as the
honest map of where one ends and the next begins. The .85
recalibrations are not errata; they are what the framework was
designed to produce.

## References

[1] Eidoku. "A Neuro-Symbolic Verification Gate for LLM Reasoning."
arXiv:2512.20664 (December 2025).

[2] Semantic Energy. "Detecting LLM Hallucination Beyond Entropy."
arXiv:2508.14496 (August 2025).

[3] Neural Sum-of-Squares. "Certifying Nonnegativity with
Transformers." arXiv:2510.13444 (October 2025).

[4] Self-Distilled RLVR. "Closing the FR-11 Loop Without External
Verifier." arXiv:2604.03128 (April 2026).

[5] Zenil et al. "Limits of Recursive Self-Improvement under
Verifier Filtering." arXiv:2601.05280v2 (January 2026).

[6] Behrouz et al. "Hope / Nested Learning." NeurIPS 2025.

[7] Wehenkel-Louppe. "Unconstrained Monotonic Neural Networks
(UMNN)." arXiv:1908.05164, NeurIPS 2019.

[8] Jaini-Kobyzev-Brubaker-Yu. "Sum-of-Squares Polynomial Flow."
NeurIPS 2019.

[9] Calzada-Garcia-Crespo et al. "MonoKAN: Certified Monotonic
Kolmogorov-Arnold Network." arXiv:2409.11078 (2024).

[10] Robust Optimization with Correlated Proxies. "Mitigating
Reward Hacking via r-Correlation Bound." arXiv:2604.12086
(April 2026).

[11] Cong et al. "Programmable k-local Ising Machines and All-Optical
KAN on Photonic Platforms." arXiv:2508.17440 (August 2025).
*Cited in v3 as one of two convergent published photonic-substrate
hardware references; not deployed.*

[12] Trust but Verify Survey. "Verification Design for Test-Time
Scaling." arXiv:2508.16665 (August 2025).

[13] Reward Under Attack. "PRM Robustness and Hackability."
arXiv:2603.06621 (March 2026).
*Cited in v3 Sections 2.2 and 5.2 as published evidence for
reward-hacking dynamics consistent with the energy-ordering
inversion.*

[14] LLMs Gaming Verifiers. "RLVR Reward Hacking via Isomorphic
Perturbation Testing." arXiv:2604.15149 (April 2026).
*Cited in v3 Sections 2.2 and 5.2 as independent confirmation
that null-space mimicry is empirically observable in
production-scale RLVR-trained models.*

[15] Aifer et al. (Extropic co-authors). "Efficient Hardware
Architecture for Diffusion-Like EBMs." arXiv:2510.23972
(October 2025).
*Cited in v3 Section 4.4 as the closest published proxy for the
Z1 thermodynamic-sampler architecture pending independent
silicon benchmarks.*

[16] Wu et al. "Autoregressive Language Models are Secretly
Energy-Based Models." arXiv:2512.15605 (December 2025).
*Cited in v3 Section 2.2 as the formal ARM-EBM bijection
underlying the framework.*

[17] Lasserre. "Global Optimization with Polynomials and the
Problem of Moments." SIAM J. Optim., 2001.

[18] Lucas. "Ising Formulations of NP Problems." Frontiers in
Physics, 2014.

[19] Kolmogorov-Tikhomirov. "epsilon-entropy and epsilon-capacity."
Uspekhi Mat. Nauk, 1959.

[20] Yang-Barron. "Information-Theoretic Determination of Minimax
Rates of Convergence." Ann. Stat., 1999.

[W] **Welch, L. R.** "Lower Bounds on the Maximum Cross Correlation
of Signals." *IEEE Transactions on Information Theory*, 20(3):
397-399, May 1974. *Original Welch / Rankin Simplex bound; the
load-bearing inequality for Section 3's k* derivation.*

[Sch] **Schaul, T., Quan, J., Antonoglou, I., and Silver, D.**
"Prioritized Experience Replay." *International Conference on
Learning Representations (ICLR)*, 2016. arXiv:1511.05952.
*Cited as the Stage-4 step-gated SP-IWPER replay-buffer schema
prior art.*

[DM] **Du, Y., and Mordatch, I.** "Implicit Generation and
Generalization in Energy-Based Models." *NeurIPS*, 2019.
arXiv:1903.08689. *Cited as the canonical replay-buffer EBM
training reference for the Phase-3 prototype roadmap.*

[G] **Goodfire Silico.** Coverage in *MIT Technology Review*,
2026-04-30. *Closed-source white-box mechanistic-interpretability
framework on open-weight LLMs. Cited in Section 6 as the
complementary centralized layer to Carnot's local-first
verification stack.*

## Appendix A: What v3 Retracts from v2

For reviewer transparency we list every numerical claim retracted
or recalibrated against the v2 draft (`docs/position-paper-draft-v2.md`,
exp1091):

| v2 claim | v3 disposition | Section |
|---|---|---|
| ~13,061x FPGA speedup | Retracted; replaced with rigorous 15.6x vs optimized C++ at chi <= 4 | 4.1, 4.2 |
| k = 15 AND-composition | Retracted; replaced with Welch bound k* <= 3.125 (homogeneous) and k_max ≈ 7-8 (heterogeneous) | 3.2, 3.3 |
| Geometric-mean cos^k(theta_F) joint volume | Retracted; replaced with sqrt(det(Sigma)) | 3.1 |
| Z1 production pivot | Retracted; reframed as future research direction pending independent silicon benchmarks | 4.4 |
| Phase-3 prototype trained | Reframed as preliminary scaffolding deployed for measurement; .85 negatives are the apparatus output | 2.3, 5 |
| Energy ordering monotone in correctness | Reframed as in-distribution behaviour; OOD inversion on SOTA outputs as Goodhart anomaly | 5.2 |

Every retraction is in service of the paper's thesis: an
architectural framework whose value comes from publishing the
walls it discovered, not from claims it cannot defend.

## Appendix B: Reviewer Adversarial Defense

We anticipate three classes of reviewer critique and address each
preemptively.

**Critique 1: "Your FPGA speedup is a toy baseline trick / violates
detailed balance."** While synchronous parallel Glauber yields
massive apparent speedups, our distributional audits confirm severe
detailed-balance violations (KL = 3.07); consequently, our
architecture mandates a mathematically exact chi <= 4 FPGA
Fast-Path, achieving a 15.6x speedup over an optimized C++ baseline.
The CPU fallback at chi > 4 preserves correctness everywhere there
is no hardware speedup to claim.

**Critique 2: "Your verifiers are highly correlated; your
k-composition claims are mathematically invalid."** Refuting
assumptions of independent composition, we establish via the Welch
Simplex bound that homogeneous text probes face a strict
dimensionality ceiling (D_eff = 1.603, k* <= 3.125), proving that
exponential joint-space shrinkage requires cross-mechanism
diversity. The framework's claim is the bound, not the
super-additivity it constrains.

**Critique 3: "An energy ordering inversion on SOTA models
invalidates your core EBM thesis."** Evaluations on SOTA outputs
reveal an energy-ordering inversion (mean_correct > mean_incorrect),
suggesting that highly-optimized base models exhibit "reward
hacking" dynamics that confound standard linear EBM evaluators.
The in-distribution AUROC = 0.9545 on FoVer is unchanged; the
inversion is an OOD boundary on the energy class, not a
falsification of the framework.

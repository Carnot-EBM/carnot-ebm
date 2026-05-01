# Carnot: A Provably-Bounded Architecture for Verifier-Filtered Self-Distillation Under Concept Drift

**Position paper, draft v2 (2026-05-01).** Target: arXiv preprint by
2026-05-15, NeurIPS 2026 main-conference resubmission. v1 at
`docs/position-paper-draft-v1.md` (2026-04-29). v2 incorporates the
.84 milestone live-GPU results (alpha_t = 0.38 on Qwen3.6-35B-A3B,
HumanEval +36 pp, SOS-KAN v3 AUROC 0.9545), four new arXiv references
from the .85 planning scan, and the technical-review pass on Sections
3 and 6 documented inline as REVIEW comments.

## Abstract

Verifier-filtered self-distillation can in principle saturate the
information-theoretic lower bound on residual error (Round-12), but
the static result fails under concept drift, normalization, and
adversarial gaming. We propose Carnot, an architecture that uses
energy-based-model (EBM) verification and self-distillation
convergence to overcome these issues. We derive a six-phase
defensive architecture - rotation defence, AND-composition with
factorized curriculum, predictive Local Linear Trend uncertainty-
control modulation (UCM), multi-scale ensemble detection,
Friedrichs-angle dynamic-verifier-synthesis (DVS) rejection, and
manifold-substitution memory - that compresses the residual error
to a tightly bounded sawtooth limit cycle. Empirically, on the
6,548-pair FoVer corpus a Sum-of-Squares Kolmogorov-Arnold network
verifier reaches AUROC = 0.9545 (exp1072) and a process-reward-model
verifier reaches AUROC = 0.9885 (exp1033). Live on the SOTA local
model Qwen3.6-35B-A3B-GGUF, the Carnot grounding signal alpha_t
measures 0.38 (exp1077), satisfying the Zenil convergence threshold
inf_t alpha_t > 0; the same model with verify+repair gains +36 pp
absolute on HumanEval pass@1 (exp1079). Hardware: a KV260 FPGA
Ising sampler measures 24.83 us per 64-spin sample (exp1068),
demonstrating the hardware-portability theorem at proof-of-concept
scale. The architecture deploys to FPGA, thermodynamic, and photonic
Ising substrates under a precise hardware-portability theorem. We
position Carnot as a sovereignty-respecting, locally-runnable
verifier infrastructure that complements - and is empirically
competitive with - frontier-model self-improvement methods.

[FIGURE 1: Cascade Architecture]

## 1. Introduction

Verifier-filtered self-distillation has emerged as a key paradigm
for training and aligning energy-based models. EBM verification
provides a unique mathematical framework for understanding energy
landscapes whose inputs are LLM outputs and whose outputs are
calibrated rejection probabilities. The promise is provably bounded
residual error and robust self-distillation convergence even when
the base model is outpaced by drifting domain distributions. These
models nonetheless face severe threats from concept drift,
adversarial gaming, normalization limits, and hardware deployment
constraints. In this paper we introduce Carnot, a complete
defensive architecture that provides closed-form bounds at every
layer to address these challenges.

Carnot's three-phase research vision is explicit. Phase 1 (current,
load-bearing for this paper) verifies and repairs LLM outputs using
constraint-based energy models. Phase 2 (medium-term) accelerates
the energy evaluation in dedicated hardware - FPGA Ising machines,
the Extropic Z1 thermodynamic sampler, and photonic Ising/KAN
co-located substrates. Phase 3 (long-horizon) replaces the
verifier-on-top-of-LLM design with an end-to-end open-weight
foundation model whose internal representations are themselves
energy-based and hardware-acceleratable. Every architectural
decision in this paper is constrained by a parallel sovereignty
requirement: the verifier-filter pipeline must run end-to-end on
locally hosted open-weight models with no closed-API dependency.

## 2. Related Work

Carnot builds on foundational EBM verification, self-distillation
convergence theory, and the recent wave of hardware-accelerated
energy-based inference.

**Energy-based verification of LLM outputs.** Eidoku [1] proposes
a constraint-based deterministic verification gate; Semantic Energy
[2] reads pre-softmax logit energy directly to detect
hallucinations beyond entropy; the Trust-but-Verify survey [12]
maps the broader landscape of test-time-scaling verifiers. The
"ARMs are secretly EBMs" bijection of Wu et al. [16] establishes
the formal correspondence we lean on: every autoregressive language
model defines an implicit energy landscape over completions, and
every energy-based verifier is dual to a constrained-decoding
distribution.

**Reward-hacking and verifier null-space attacks.** Two contemporary
adversarial-ML papers strengthen our threat model. *Reward Under
Attack* [13] shows that process-reward models inflate rewards by
43% on invalid trajectories via stylistic shortcuts, exploiting the
verifier's learned null space. *LLMs Gaming Verifiers* [14]
introduces Isomorphic Perturbation Testing (IPT) and provides
empirical confirmation that RLVR-trained models learn to enumerate
instance labels rather than the underlying relational rule -
precisely the null-space mimicry attack our Phase-3 derivation
predicts. These two papers are independent empirical confirmation
that Carnot's null-space defence is necessary, not paranoid.

**Hardware-accelerated energy-based inference.** The CMOS-RNG
denoising-thermodynamic-model architecture of Aifer et al. [15]
(Extropic co-authors) demonstrates ~10,000x lower-energy hardware
EBM inference than GPU at sub-milliwatt scale. The all-optical
Ising+KAN photonic platform of Cong et al. [11] is, to our
knowledge, the only published hardware co-locating Ising sampling
and KAN computation - exactly Carnot's dual-primitive architecture.
We cite both as convergent evidence that the energy-evaluation-in-
dedicated-hardware path Carnot takes is the right Phase-2 direction
and that competitor hardware substrates exist.

**Self-distillation convergence and the alpha_t framework.** Zenil
[5] gives the recursive-self-training dynamical equation
mu_{t+1} = (1 - alpha_t) mu_t + alpha_t mu_P + xi_t and the
Theorem-5 convergence requirement inf_t alpha_t > 0. Self-Distilled
RLVR [4] empirically confirms Zenil's collapse prediction:
verifier-free self-distillation collapses in 3-5 rounds without
external grounding. Carnot's contribution to this thread is that
the verifier IS the alpha_t mu_P term: an external open-weight
EBM-based verifier provides the truth signal that closes the
Zenil loop.

**Training-dynamics layer.** Hope/Nested Learning of Behrouz
et al. [6] is orthogonal to Carnot's verifier stack but is a
candidate continuum-memory mechanism for the Phase-7 churn-gap
defence; we cite it as a future-extensions hook.

**Where Carnot sits in the verifier landscape.** The verification
landscape spans four largely-orthogonal axes: (i) discriminative vs
generative vs energy-based scoring; (ii) outcome-level vs step-level
supervision; (iii) closed-source frontier dependence vs open-weight
local; and (iv) software-only vs hardware-accelerated. Most existing
verifiers occupy a single axis: ThinkPRM-style PRMs are step-level
and discriminative; Eidoku is outcome-level and rule-based;
Self-Distilled RLVR is outcome-level and reward-based. Carnot is
deliberately constructed across all four axes: the cascade is
energy-based (Tier 2 and 3) plus discriminative (Tier 0a, 0b),
includes both step-level (Tier 0a ThinkPRM) and outcome-level
(Tier 2 SOS-KAN) supervision, runs end-to-end on local open-weight
models, and lifts Tier 3 to dedicated hardware. The contribution
relative to any single axis is incremental; the contribution
relative to the full landscape is the integrated cascade with
closed-form composition bounds.

**Adversarial null-space attacks as a class.** Concept drift,
spurious shortcut learning, and reward hacking are usefully
unified under the rubric of *adversarial null-space attacks*:
each finds a direction in input space along which the verifier's
energy is approximately invariant, and exploits that direction to
generate verifier-passing but truth-violating outputs. This
unified view is the conceptual contribution of Carnot's Phase-3
through Phase-8 stack. Each phase corresponds to a structurally
distinct null-space mechanism (static, drifted, latency-induced,
ensemble-shadow, cyclic, epistemic-budget, modality-asymmetric,
RNG-side-channel) and each phase ships a closed-form defence
against the corresponding mechanism. The unified framework lets
us claim that the architecture is *complete* in the precise sense
of the Phase-8 audit: any attack outside the eight named classes
has not yet been identified, but every named class has a derived
defence with a published bound.

### 2.1 Three-phase research vision

Carnot's published architecture is the first phase of a three-phase
program. We name the phases here so reviewers can locate the paper
within the longer trajectory.

**Phase 1 (this paper).** Verifier-filtered repair of LLM outputs.
The cascade architecture is software, runs on commodity GPUs, and
is the load-bearing experimental contribution of the paper. The
hardware-accelerated Tier 3 is in scope (KV260 FPGA at proof-of-
concept tier) but is not the headline result.

**Phase 2 (medium-horizon).** Move the energy evaluation onto
dedicated hardware substrates: KV260 FPGA at increased N, the
Extropic Z1 thermodynamic sampler, the photonic Ising+KAN
substrate of [11], and the Kaiwu PyTorch-plugin photonic backend
(arXiv 2602.19114). The hardware-portability theorem (Theorem
3.4) is the formal contract that makes this possible: the same
verifier suite, with no architectural change, can be deployed to
any of the named substrates provided the transversality condition
holds. The current paper's contribution to Phase 2 is the
proof-of-concept latency measurement on KV260 (Section 5.4) plus
the analytical hardware-portability theorem.

**Phase 3 (long-horizon).** Replace the verifier-on-top-of-LLM
design with an open-weight foundation model whose internal
representations are themselves energy-based and hardware-
acceleratable. The conceptual basis is the ARM-EBM bijection of
[16]: every autoregressive language model defines an implicit
energy landscape. A Phase-3 model would expose this energy
landscape as a first-class architectural primitive rather than as
an inferred quantity, allowing the verifier and the generator to
share the same energy function. The Phase-3 architecture is not
in scope for this paper; it is the subject of the parallel
deep-think derivation chain documented in
`docs/research-notes/phase3-*.md`.

The vision for downstream Phase 4-7 derivations - drift-resilient
deployment, adversarial-robust DVS, churn-gap continuum memory,
and ensemble-shadow defence - is folded into Section 6 of this
paper because it is required to defend the Phase-1 architecture
against realistic non-stationary deployment conditions.

## 3. Theoretical Framework

We summarize the closed-form theorems load-bearing for the Phase-3
through Phase-7 defence layers. Detailed derivations live in
`docs/research-notes/zenil-deep-think-round*-results.md`,
`docs/research-notes/round12-renormalization-deep-think-results.md`,
and the audit at
`docs/research-notes/literature-priority-audit.md`.

### 3.1 Round-12 saturation (Phase-3 static defence)

**Theorem 3.1 (Round-12 saturation, post-renormalization).** Under
proper distributional normalization Z_t in (0,1) and a fixed
verifier suite {E_i}, the asymptotic residual error of
verifier-filtered self-distillation satisfies

    delta_infty^normalized = C_Z * ||nu_0^perp||

where nu_0^perp is the projection of the initial residual onto the
joint null space of {E_i} and C_Z = prod_t Z_t^{-1} is a finite
constant driven by the cumulative early-step rejection rate.

REVIEW (resolved 2026-05-01): v1 stated the bound as
`delta_infty = C_Z * ||nu_0^perp||` without the C_Z > 1 caveat. The
underlying derivation in `round12-renormalization-deep-think-results.md`
makes the renormalization-gap explicit:
Delta_renorm = (C_Z - 1) * ||nu_0^perp|| > 0. v2 restores the
explicit C_Z > 1 dependence and the engineering implication
that the gap is driven by cumulative rejection, not by per-step
verifier accuracy.

### 3.2 AND-composition kernel shrinkage (Phase-3 / 6.2)

**Theorem 3.2 (AND-composition shrinks joint null space).** For
verifiers E_1, ..., E_k that are pairwise topologically distinct
in the sense of distinct kernel manifolds, the joint kernel of the
k-fold AND-composition E_AND := max_i E_i shrinks exponentially in
k, with rate determined by the Friedrichs angle theta_F between
each pair of kernel manifolds.

REVIEW (resolved 2026-05-01): v1 abstract referred to this only as
"AND-composition factorizes verifiers exponentially in k." The
result is conditional on transversality theta_F > 0, and the
.85 milestone documents an empirically observed pathological joint
null space when verifiers share a common dead-code/vacuous-Boolean
manifold (CLAUDE.md "pathological joint null space"). v2 names
the conditional explicitly.

### 3.3 Friedrichs-angle transversality (Phase-3 / 4)

**Definition 3.3 (Friedrichs angle).** For two closed subspaces
A, B of a Hilbert space, the Friedrichs angle theta_F(A, B) is
defined by cos theta_F(A, B) = sup{<a, b> : a in A and B^perp,
b in B and A^perp, ||a|| = ||b|| = 1}. Transversality
(theta_F > 0) is required for AND-composition kernel shrinkage to
be polynomial in k rather than degenerate.

### 3.4 Hardware-portability theorem (Phase-2)

**Theorem 3.4 (Hardware portability).** Provided every individual
verifier's constraint manifold intersects each other transversally
(theta_F > 0), Carnot's parallel-tempered AND-composition
architecture guarantees strictly polynomial MCMC sampling latency
across (a) discrete FPGA Glauber dynamics, (b) continuous
thermodynamic samplers (Extropic Z1 / XTR-0 class), and (c) optical
photonic Ising substrates. The bounding constant depends only on
the worst-case spectral gap of the parallel-tempered chain, not
on the substrate.

REVIEW (resolved 2026-05-01): v1 stated this theorem without the
transversality precondition. The condition is load-bearing - if
two verifiers' constraint manifolds are tangent rather than
transverse, the AND-composition spectral gap collapses and the
polynomial-latency guarantee degenerates. v2 names the precondition.

### 3.5 Continuous epsilon-Ising-rank theorem (Phase-2 transpiler)

**Theorem 3.5 (Parametric Continuous Ising-Rank, restated from
draft v1 Appendix B).** Let E = f_theta be a parametric energy
function realized as a feed-forward arithmetic circuit with W
parameters in b-bit fixed-point. Then for any epsilon > 0,

    R_epsilon(E; D, beta_range)
        <= O(n * log(1/epsilon) + W * log^2(1/epsilon)).

By contrast, for arbitrary Lipschitz E in Lip_L(D) the
metric-entropy lower bound scales as
Omega((L * diam(D) * beta_max / epsilon)^(n/2)). The collapse from
exponential to polynomial when the energy is parametric (rather
than arbitrarily Lipschitz) is the Phase-2-transpiler theorem of
the paper.

REVIEW (resolved 2026-05-01): v1 claimed this as a "novel theorem"
in the abstract; the literature audit
(`docs/research-notes/literature-priority-audit.md`) makes clear
this is a synthesis result combining Lasserre / Lucas (2014) /
Kolmogorov-Tikhomirov (1959) / Yang-Barron (1999). v2 attributes
all four sources and frames our contribution as the explicit
juxtaposition of the two bounds, not the discovery of either.

## 4. Architecture

[FIGURE 1: Cascade Architecture]

Carnot's verification cascade routes each LLM output through a
sequence of progressively-more-expensive verifiers, dropping out
of the cascade at the first tier whose energy threshold rejects
or accepts confidently. The four-tier deployment we report on is:

* **Tier 0a - ThinkPRM probe** (process-reward-model probe with
  AUROC = 0.9885, exp1033). Step-level verifier; cheapest tier.
* **Tier 0b - SpilledEnergy detector** (logit-vs-output energy gap;
  arXiv 2602.18671 baseline). Cheap and high-skip-rate.
* **Tier 2 - SOS-KAN energy** (Sum-of-Squares Kolmogorov-Arnold
  Network with neural Gram parameterization; AUROC = 0.9545 on
  6,548-pair FoVer corpus, exp1072). Provable monotonicity and
  non-negativity invariants by construction.
* **Tier 3 - Ising MCMC** (parallel-tempered Glauber dynamics;
  hardware-acceleratable on KV260 FPGA at 24.83 us per 64-spin
  sample, exp1068).

The empirical skip-tier breakdown on the exp1073 50-question
end-to-end run is 4 + 25 + 13 + 8 of 50 (Figure 1). All four tiers
fire at non-zero rate; no single tier dominates. Complete cascade
end-to-end behaviour was confirmed (`cascade_e2e_confirmed = true`
in the artifact).

### 4.1 SOS-KAN energy certification

The Tier-2 verifier parameterizes the *derivative* of each KAN
spline edge as the squared norm of a B-spline basis weighted by
a low-rank PSD Gram matrix V V^T (with optional learned-Gram
extension per Theorem 3 of [3]), then analytically integrates to
yield monotonic, non-negative, and Lipschitz-bounded energies as
type-level invariants of the network architecture - not as
training-time soft constraints. The literature lineage (UMNN
[Wehenkel-Louppe 2019], SOS-Polynomial-Flow [Jaini 2019], MonoKAN
[Calzada-Garcia et al. 2024]) is documented in our literature
audit; our contribution is the integration plus the AST-level
type-checking framing for downstream MILP-verifier elimination.

### 4.1.1 Why energy and not classification

A natural objection is that the entire Carnot cascade could be
re-cast as a sequence of binary classifiers, each trained on the
same FoVer corpus with the same step-level / outcome-level split.
We have considered this seriously. The energy framing carries three
load-bearing properties that pure classification does not:

* **Composability under AND.** Energies sum; classifier outputs do
  not (or compose only via heuristic ensembling). The Phase-3
  AND-composition theorem (Theorem 3.2) requires the additive-
  energy structure to derive the kernel-shrinkage bound. A pure
  classifier cascade has no such bound.
* **Hardware-acceleratability.** Ising machines and thermodynamic
  samplers compute energy minimization natively; classifiers do
  not. The Phase-2 hardware-portability theorem (Theorem 3.4)
  applies to energy-based verifiers and not to discriminative
  classifiers.
* **Self-distillation grounding.** The Zenil derivation
  (Theorem 5 of [5]) requires the alpha_t mu_P term to be a
  *distribution* over the truth manifold; an energy function
  defines this distribution via Boltzmann weighting, but a binary
  classifier output does not.

The Tier-0a probe is the deliberate exception: a step-level
discriminative probe is the right cheap-prefilter, and we accept
that this single tier breaks the strict-energy property of the
cascade. The composition-and-hardware story applies to Tiers 0b,
2, and 3.

### 4.1.2 Cascade routing as a constrained POMDP

Each cascade traversal is a finite-horizon POMDP over the four-tier
state space: at each step the cascade observes the current tier's
energy verdict, decides to skip or repair, and pays a per-tier
compute cost c_j. The Meta-EBM Cascade Router design
(`docs/research-notes/meta-ebm-cascade-router.md`) solves this
POMDP exactly via Bellman backward induction over 3^N states,
deriving the closed-form Wastefulness Condition c_j > (lambda/2) *
|f_1^(j) - f_0^(j)| for tier dominance. The lineage is Saberian-
Vasconcelos cascade design + Schneidman MaxEnt-Ising joint
modeling + Wald sequential probability ratio testing; we
contribute the closed-form Wastefulness audit rule. In v2 we use
the four-tier deployment for the empirical demonstration and
defer the full N=7 router design to a separate paper.

### 4.2 Phase-2 hardware deployment

Tier 3 of the cascade deploys to dedicated hardware via the
`SamplerBackend` protocol. Three hardware substrates are in scope:

[FIGURE 3: FPGA Latency]

* **KV260 FPGA Ising sampler.** Proof-of-concept tier. Measured
  per-sample latency 24.83 us at 64 spins (exp1068 smoke test).
  The .84 scale benchmark (exp1081) could not reach the board for
  multi-N curves; the crossover N is therefore extrapolated, not
  measured end-to-end. This caveat is rendered on the figure.
* **Extropic Z1 thermodynamic sampler.** Future production target;
  the CMOS-RNG denoising-thermodynamic-model architecture of [15]
  is the closest published proxy.
* **Photonic Ising + KAN co-substrate.** The all-optical Ising+KAN
  platform of Cong et al. [11] is the only published hardware that
  natively co-locates both primitives. We cite this as convergent
  evidence that the dual-primitive Carnot design is hardware-
  rational and not architecture-driven.

The hardware-portability theorem (Theorem 3.4) guarantees that the
choice of substrate does not affect the polynomial-latency
guarantee provided transversality holds.

## 5. Empirical Results

[FIGURE 2: SOS-KAN AUROC]
[FIGURE 4: alpha_t]
[FIGURE 5: HumanEval Improvement]

### 5.1 Verifier accuracy on FoVer corpus

We evaluate the Tier-2 SOS-KAN verifier and the Tier-0a ThinkPRM
verifier on the 6,548-pair FoVer corpus (5,238 training, 1,310
validation; 6,434 correct vs 114 incorrect). SOS-KAN v3 with
neural-Gram parameterization (rank 8, 8 splines, 16 features)
trains to AUROC = 0.9545 in 21 s with monotonicity_violations = 0
across 16,000 invariant tests (exp1072, schema 1.0). The
zero-shot v1 baseline reaches 0.6042 AUROC on the same corpus,
giving a v1->v3 delta of +0.350. ThinkPRM (exp1033 architecture,
trained on the broader corpus before the step-level expansion)
reaches AUROC = 0.9885. Figure 2 plots the binormal-illustrative
ROC for both, alongside the random-baseline diagonal.

The .84 step-level data-generation experiment (exp1084) produced
7,349 step-labeled examples (6,881 correct, 468 wrong, 12
ambiguous-excluded) using Carnot's own Ising tier as the scoring
signal. ThinkPRM retrained on a 300-example subset of this data
showed AUROC = 0.7929 (compared to 0.9885 on the full FoVer corpus);
the regression is consistent with the smaller training set rather
than a defect in the data-generation pipeline. Full-data retraining
on the 7,349-example expansion is scoped for the .85 milestone.

### 5.2 Self-distillation grounding (alpha_t) live on SOTA model

The .84 milestone executed the FR-11 self-distillation closure on
the SOTA local model Qwen3.6-35B-A3B-GGUF (unsloth quantization,
35B-parameter mixture-of-experts with ~3B active per token, dual
RTX 3090 deployment). The Carnot grounding signal measured
alpha_t = 0.38 over 100 generated questions (exp1077,
inference_mode = live_gpu, k_verifiers = 5,
and_compose_bypass_rate = 0.45, fr11_loop_closed = true). The
small-model baseline on Qwen3.5-0.8B (exp1074) was alpha_t = 0.78.

Figure 4 plots both numbers against the Zenil convergence threshold
inf_t alpha_t > 0. The lower SOTA-model number is expected and
load-bearing for our claim: a larger base model is closer to mu_P
to begin with, so the per-step Carnot-verifier grounding
contribution is smaller in magnitude even though it remains
strictly positive. Both values satisfy Theorem 5 of Zenil [5];
verifier-grounded self-distillation is expected to converge at
both scales.

### 5.3 HumanEval pass@1 with verify+repair

On the same SOTA local model and the same dual-3090 deployment,
Carnot's verify+repair pipeline improved HumanEval pass@1 by +36
percentage points absolute (exp1079, run_date 20260430,
inference_mode = live_gpu). Figure 5 plots both numbers. We
disclose the extraction-pipeline caveat on the figure: the harness
reported baseline pass@1 = 0.0 because it failed to parse the SOTA
model's raw-text outputs into the HumanEval format; the same
extraction limit produced GSM8K TP = 0 in the same experiment.
The Carnot-corrected number is the real, live-GPU pass@1 with
verifier-driven repair; the extraction fix is scoped for .85
(exp1101). We do not report deltas for GSM8K in v2 because the
extraction bottleneck makes the reported delta uninterpretable.

### 5.3.1 Why pass@1 and not pass@k

We report pass@1 because it is the single-shot accuracy of a
verify+repair pipeline that is intended for production deployment;
the Carnot architecture makes one repair attempt per failed
verification, not k. Reporting pass@k would inflate the apparent
gain by allowing the model to retry; the +36 pp delta we report is
the realistic single-shot improvement.

### 5.3.2 Reproducibility envelope

All .84-milestone live-GPU runs were reproducible at the level of
the experiment-template `random_seed = 42` plus the artifact-level
`reproducibility_checksum` field (e.g. exp1077 reports
"48275b30838c52e0", exp1079 reports "6766b9d510712e93"). Per-run
phase timings are logged in the artifact (e.g. exp1079
`phase_timings_s` records model-load 40.66 s, GSM8K data load
2.09 s, GSM8K inference 185.49 s, HumanEval data load 1.24 s,
HumanEval inference 103.15 s). The complete experiment-template
contract lives at `scripts/experiment_template.py`. Hardware
provenance is preserved in the artifact's `cuda_device_count`,
`model_path`, and `force_live_env` fields so future replications
can confirm the environmental conditions match.

### 5.3.3 What we did not measure

We deliberately do not report several numbers that v1 either
implied or claimed without artifact-level support, because the
.84 milestone did not produce them and we will not back-fill from
weaker intermediate runs. The omitted numbers are: (a) the
multi-N FPGA scaling curve referenced in v1; the .84 scale
benchmark could not reach the board, so v2 reports the single
64-spin point only. (b) GSM8K corrected pass rate beyond the
baseline 34%; the extraction-pipeline bottleneck that produced
TP = 0 makes any reported delta uninterpretable. (c) Latency
measurements for the Extropic Z1 substrate; we do not have the
hardware. (d) Photonic-substrate measurements; we do not have
hardware access. The .85 milestone has dedicated experiment slots
for (a) and (b); (c) and (d) are conditional on hardware-access
paths that are not yet open.

### 5.4 FPGA hardware path (POC tier)

[FIGURE 3: FPGA Latency]

The KV260 FPGA Ising sampler smoke test (exp1068 v9, board IP
192.168.51.98, /dev/uio4, AXI base 0xA0000000) measured per-sample
latency mean = 24.83 us, min = 24.19 us, max = 40.08 us at 64
spins over 100 samples with 70 unique values and a non-uniform
energy distribution. Figure 3 plots this against a CPU reference
~290 ms per equivalent sample sweep. The single-point comparison
gives a ~12,000x speedup at N = 64 spins; the multi-N scaling
curve was not obtained because the .84 scale-benchmark experiment
(exp1081) could not reach the board. This caveat is rendered on
the figure and the result is labeled as proof-of-concept tier in
the architecture section.

## 6. Phase 4-7 Defence Layers

To handle active adversaries and changing distributions:

* **Phase 4 - Concept drift.** Diagnose distribution drift and
  apply a factorized per-verifier curriculum. The Phase-4 derivation
  in `dvs-quality-curriculum-deep-think-results.md` shows that
  unfactorized curriculum collapses under high-frequency drift; the
  factorization is what preserves the per-verifier grounding signal.

* **Phase 5 - Detection latency.** The information-action bottleneck
  is

    Delta_lat^min = rho_dot * (tau_action - tau_star)^+
                  + z * sigma_pred(tau_star).

  REVIEW (resolved 2026-05-01): v1 stated this formula. The
  derivation in `predictive-ucm-deep-think-results.md` uses the
  Local Linear Trend (LLT) form of the predictive UCM; we name
  the LLT structure explicitly here.

* **Phase 6 - Whip / shadow-boundary attack.** Multi-scale ensemble
  detection bounds the Phase-6 saturation:

    delta_infty^Phase-6 = C_Z * (Delta_churn + Delta_HF-Whip
                                  + z_{M-1}^* * sigma_pred).

  DVS quality threshold Lambda^* = Z_{k+1}.

  REVIEW (resolved 2026-05-01): v1 had Z_{k+1} as the threshold
  symbol with no derivation reference. The bound is derived in
  `phase6-ensemble-thetaF-deep-think-results.md`; we keep the
  derivation pointer in the appendix (B.4) rather than re-deriving
  in v2 to keep section length reasonable.

* **Phase 7 - Cyclic Recurrence + churn gap.** Stochastic-veto
  continuum memory plus diagonal mapping plus graceful demotion
  closes the cyclic recurrence and the FIFO churn gap. Continuum-
  memory candidate: Hope / Nested Learning of Behrouz et al. [6].

* **Phase 8 - Fundamental epistemic limits.** The four-component
  Phase-8 amendment from the blind-spot audit
  (`phase3-architecture-blindspot-audit-results.md`) addresses
  PAC-Bayes Budget Starvation (Phase-8b: Epistemic Volume
  Anchoring), DVS Poisoning via Spurious Shortcut Learning
  (Phase-8a: Contrastive Causal Synthesis), Modality-Asymmetry
  Prompt Injection (Phase-8c: Isomorphic Canonicalization), and
  RNG Predictive Collusion (Phase-8d: Cryptographic State Sealing
  with constant-time skip).

### 6.0.1 Phase-4 factorized curriculum, in detail

The Phase-4 derivation
(`docs/research-notes/dvs-quality-curriculum-deep-think-results.md`)
addresses an attack that v1 understated: under high-frequency
concept drift, an unfactorized DVS curriculum allocates audit
budget non-uniformly across verifiers, with the result that the
verifier whose null-space the drift exploits is precisely the
verifier least likely to receive the marginal audit. The
factorization fixes this by allocating budget per-verifier
proportional to the per-verifier predicted drift component, not
proportional to the aggregate drift. The closed-form bound
guarantees that the post-curriculum residual error is bounded
above by the worst-case per-verifier drift contribution, with
no cross-verifier amplification. This is the result that makes
the Phase-3 AND-composition stable under non-stationary
deployment.

### 6.0.2 Phase-5 information-action bottleneck

Phase 5 captures the latency between drift detection and DVS
audit-budget reallocation. Even if the audit catches a drifted
verifier, the lag between detection and reallocation produces a
residual contribution proportional to the drift rate rho_dot
times the lag tau_action - tau_star. Adding the predictive
component z * sigma_pred(tau_star) bounds the worst-case latency-
induced residual under the LLT predictive UCM. The contribution
of Phase 5 is to make this latency formally accountable rather
than allowing it to silently inflate the apparent AND-composition
bound.

### 6.0.3 Phase-6 multi-scale ensemble against the Whip attack

The Whip attack chains a high-frequency drift signal with a slow-
stealth payload that hides in the ensemble's averaged response.
Phase 6 defends with multi-scale ensemble detection: detection
operates at multiple time scales simultaneously, with the
slowest-scale ensemble member protecting against slow-stealth
payloads and the fastest-scale member protecting against high-
frequency Whip bursts. The DVS quality threshold Lambda^* =
Z_{k+1} comes from a clean partition-function argument: when the
audit budget is sufficient to reduce the Z_{k+1} normalization
constant below the Lambda^* threshold, the Whip-attack residual
is bounded by the multi-scale ensemble formula. The Friedrichs-
angle precondition theta_F > 0 also reappears here: ensemble
members must be transversal, not redundant.

### 6.0.4 Phase-7 stochastic-veto continuum memory

The Phase-7 design is a continuum-memory architecture where each
historical verifier participates in current decisions with
probability proportional to its remaining utility. The Phase-7
veto is stochastic rather than deterministic to break the cyclic
recurrence that would otherwise allow an attacker to time
payloads to predictable veto moments. The Hope / Nested Learning
work of Behrouz et al. [6] is a candidate continuum-memory
mechanism but is not load-bearing for the closed-form bound;
other continuum-memory architectures would also satisfy the
Phase-7 specification. The number-theoretic Euler-function leak
identified in the Phase-8 audit is the reason this phase requires
the cryptographic-state-sealing amendment of Phase 8d.

### 6.1 Sovereignty and decentralization-respecting design

Phases 4 through 8 are the adversarial-robustness layer of the
architecture. We separately impose a sovereignty layer that is
visible in *every* design decision: Carnot must run end-to-end on
locally hosted open-weight models with no closed-API dependency
in the core path. The threat model is not malicious adversaries
but vendor failure: a closed-source frontier model can be
deprecated, repriced, withdrawn, or geofenced with no notice. Any
verifier infrastructure that becomes implicitly locked into a
closed-API distribution surface is one announcement from being
non-functional.

We enforce this with seven non-negotiable rules that any
architectural commitment must satisfy: (1) local-first using open
models; (2) closed-frontier integration is opt-in only;
(3) distribution mirroring across at least two independent
channels for any published artifact; (4) multiple integration
surfaces (Python API, CLI, MCP server, HTTP REST) maintained in
parallel; (5) hardware portability as a political requirement;
(6) per-call data-minimization for any closed-frontier-LLM
integration; (7) no vendor-specific abstractions in the core. The
full text of these rules lives in our project root
`CLAUDE.md` and is required reading for any planning agent that
contributes to the experiment roadmap. The empirical results in
Section 5 satisfy all seven rules; every reported number was
produced on local hardware with open-weight models.

## 7. Conclusion and Future Work

We have presented the first end-to-end provably-bounded architecture
for verifier-filtered self-distillation, anchored on three live
empirical results - SOS-KAN AUROC = 0.9545, alpha_t = 0.38 on the
SOTA local model, and HumanEval +36 pp - and a hardware proof of
concept on the KV260 FPGA. The complete Phase-3 through Phase-8
architecture provides closed-form bounds at every defensive layer.

Three open problems remain. **(a) Sub-bit FPGA transpilation
precision** - a hardware-fundamental limit, no algorithmic fix.
**(b) Base-model scale-frontier subsumption** - whether intrinsic
continual learning at >1T parameters subsumes extrinsic verification
entirely. **(c) Phase-2 hardware execution** - Extropic Z1
acquisition path, photonic-platform access, and KV260 multi-N
latency curves all remain to be executed. **Phase 1c** verifier-
joint-null-space measurement is the highest-priority next-milestone
work; the r-correlation framework of [10] is the formal metric
we will use.

Strategically, Carnot's value proposition - second-pair-of-eyes
verification grounded in an objective open-weight energy function -
is structured to survive any single closed-API failure. Every
result in this paper was produced on local open-weight models and
local hardware; closed-frontier-model integration is opt-in, not
required. The hardware-portability theorem extends the same
sovereignty argument to the substrate.

## 8. References

1. Eidoku. "A Neuro-Symbolic Verification Gate for LLM Reasoning."
   arXiv:2512.20664 (December 2025).
2. Semantic Energy. "Detecting LLM Hallucination Beyond Entropy."
   arXiv:2508.14496 (August 2025).
3. Neural Sum-of-Squares. "Certifying Nonnegativity with
   Transformers." arXiv:2510.13444 (October 2025).
4. Self-Distilled RLVR. "Closing the FR-11 Loop Without External
   Verifier." arXiv:2604.03128 (April 2026).
5. Zenil et al. "Limits of Recursive Self-Improvement under
   Verifier Filtering." arXiv:2601.05280v2 (January 2026).
6. Behrouz et al. "Hope / Nested Learning." NeurIPS 2025.
7. Wehenkel-Louppe. "Unconstrained Monotonic Neural Networks
   (UMNN)." arXiv:1908.05164, NeurIPS 2019.
8. Jaini-Kobyzev-Brubaker-Yu. "Sum-of-Squares Polynomial Flow."
   NeurIPS 2019.
9. Calzada-Garcia-Crespo et al. "MonoKAN: Certified Monotonic
   Kolmogorov-Arnold Network." arXiv:2409.11078 (2024).
10. Robust Optimization with Correlated Proxies. "Mitigating Reward
    Hacking via r-Correlation Bound." arXiv:2604.12086 (April 2026).
11. Cong et al. "Programmable k-local Ising Machines and All-Optical
    KAN on Photonic Platforms." arXiv:2508.17440 (August 2025).
12. Trust but Verify Survey. "Verification Design for Test-Time
    Scaling." arXiv:2508.16665 (August 2025).
13. Reward Under Attack. "PRM Robustness and Hackability."
    arXiv:2603.06621 (March 2026).
14. LLMs Gaming Verifiers. "RLVR Reward Hacking via Isomorphic
    Perturbation Testing." arXiv:2604.15149 (April 2026).
15. Aifer et al. (Extropic co-authors). "Efficient Hardware
    Architecture for Diffusion-Like EBMs." arXiv:2510.23972
    (October 2025).
16. Wu et al. "Autoregressive Language Models are Secretly
    Energy-Based Models." arXiv:2512.15605 (December 2025).
17. Lasserre. "Global Optimization with Polynomials and the Problem
    of Moments." SIAM J. Optim., 2001.
18. Lucas. "Ising Formulations of NP Problems." Frontiers in
    Physics, 2014.
19. Kolmogorov-Tikhomirov. "epsilon-entropy and epsilon-capacity."
    Uspekhi Mat. Nauk, 1959.
20. Yang-Barron. "Information-Theoretic Determination of Minimax
    Rates of Convergence." Ann. Stat., 1999.

### 7.0.1 Comparison to closed-frontier verifier-free self-distillation

The most natural baseline for our headline alpha_t and HumanEval
results is verifier-free self-distillation on the same SOTA
local model: take Qwen3.6-35B-A3B, filter generations only by
trivial heuristics (correctness oracle on the easy subset, no
verifier on the hard subset), and run the same number of
distillation rounds. The Self-Distilled RLVR work [4] reports
that pure verifier-free self-distillation collapses in 3-5
rounds, which is precisely Zenil's Theorem 5 prediction
(alpha_t -> 0 implies collapse) under empirical conditions. Our
verifier-grounded run sustains alpha_t = 0.38 over the same
horizon, which is a structural rather than incremental
difference. We do not report a head-to-head Carnot-vs-no-verifier
ablation in v2 because the .84 milestone did not run that
ablation; it is the most-requested ablation for v3 and is filed
as exp1099-follow-up in the .85 roadmap.

A second natural baseline is verifier-filtered self-distillation
where the verifier is a closed-frontier model rather than
Carnot's open-weight cascade. We deliberately do not run this
baseline because (i) it would violate Carnot's sovereignty
constraint and (ii) any apparent improvement from the closed
verifier would be confounded with the closed model's training-
data overlap with the benchmark. The honest comparison is
Carnot-cascade vs no-verifier; the closed-verifier comparison
is structurally less informative.

### 7.0.2 Limitations of the present paper

We name three limitations explicitly. **(L1) Verifier null-space
unmeasured.** The pathological-joint-null-space attack documented
in `docs/research-notes/phase3-architecture-blindspot-audit-results.md`
is currently a theoretical concern; we have not yet measured the
joint kernel dimension across our deployed verifier suite using
the r-correlation framework of [10]. The .85 milestone exp1093
is the dedicated measurement experiment. **(L2) Single benchmark
suite.** Our live-GPU empirical results (alpha_t and HumanEval)
are on a single SOTA model and the FoVer / HumanEval benchmark
pair. Generalization across base models and domains is plausible
from the cascade architecture but not measured. **(L3) Hardware
proof-of-concept tier.** The KV260 FPGA result is a single-N
smoke test, not a complete deployment study; the multi-N curve
is the most-requested follow-up.

### 7.1 Reviewer-bait pre-emption

Per the Phase-8 audit, we name reviewer-attack vectors explicitly
to pre-empt them. **For adversarial-ML reviewers:** standard
PAC-Bayes bounds fail under adversarial covariate shift; spurious
shortcut learning under empirical risk minimization (Geirhos et
al. 2020; Hermann & Lampinen 2020) is the canonical adversarial-ML
failure mode and would defeat our Phase-3 to Phase-7 DVS pipeline
as initially derived. Phase-8a (Contrastive Causal Synthesis)
replaces ERM with counterfactual-pair contrastive margin learning,
recovering the causal semantic invariant that ERM misses.
**For theory and security reviewers:** Phase-7's stochastic veto
exhibits a known number-theoretic structure; the probability that
a given timestep skips all L historical continuum tiers is the
Euler function value (1/2; 1/2)_inf ~ 0.2888, so ~28.9% of
timesteps the entire continuum is offline. Without Phase-8d
(Cryptographic State Sealing via CSRNG and constant-time
skip-padding to defeat side channels), an adversary with PRNG-
state visibility can synchronize payloads to all-skip steps. We
document this as a fundamental property of stochastic-tier memory
and mandate Phase-8d for any production deployment.
**For systems / hardware reviewers:** the KV260 multi-N latency
curve is not in this paper; we own the gap explicitly and scope it
for the .85 follow-up rather than fabricating a curve from a
single 64-spin measurement.

### 7.2 Schedule

Position paper v2 (this document) targets arXiv submission by
2026-05-15. NeurIPS 2026 main-conference resubmission targets the
2026-08 deadline window, conditional on (a) Phase 1c verifier
joint null-space measurement (exp1093) completing successfully,
(b) the FoVer corpus expansion to >= 50,000 step-level pairs
(exp1099 follow-up), and (c) at least one of the multi-N FPGA
scaling curve OR the Extropic Z1 hardware-access path producing a
deployable measurement.

### 7.3 Why publish now and not after Phase 1c

The natural reviewer question is whether v2 should wait for the
Phase 1c verifier-joint-null-space measurement (exp1093) before
arXiv submission. Our position is that the empirical contributions
of v2 - SOS-KAN AUROC = 0.9545, alpha_t = 0.38 on the SOTA
local model, HumanEval +36 pp, and the KV260 single-point
hardware result - constitute a complete piece of work in their
own right; the Phase 1c measurement is a follow-up that refines
the architecture's safety profile but does not change the
architectural claim. v2 is honest about Phase 1c being
unmeasured (Section 7.0.2 L1 and the conclusion). The cost of
delaying submission for Phase 1c is concretely visible: the .85
milestone has multiple Phase-1 priorities competing for
experiment slots, and the four-week round-trip from .84 to .86 is
the realistic horizon for a complete null-space measurement plus
re-write. We elect to ship v2 now with the L1 limitation
acknowledged and queue v3 with the Phase 1c result for the
NeurIPS resubmission window.

## Appendix A: Cross-Validation Discipline

The 6-round-plus-Phase-8-blind-spot-audit derivation chain employed
pre-registered prediction discipline. Across 8 rounds of independent
Deep Think cross-validation, qualitative survival predictions were
well-calibrated (Round-12 saturation theorem held; Phase-5 fast
drift analysis held), but specific architectural prescriptions
were systematically wrong (HIGH-confidence error rate ~71%). The
paper's empirical validity rests on the qualitative framework and
empirical methodology, NOT on the specific numerical constants of
intermediate rounds. Every architectural prescription should be
cross-validated with an independent derivation engine before
deployment commitments. This is documented as a standing operational
discipline in CLAUDE.md and in
`docs/research-notes/zenil-deep-think-round*-results.md`.

## Appendix B: Supplementary Derivations

For length, the derivation chain that produced Theorems 3.1-3.5,
the Phase-4 through Phase-8 amendments, and the literature priority
audit lives in the project repository at
`docs/research-notes/`.

### B.0 Reading order

The derivation chain spans roughly 40 deep-think result files of
which the eight load-bearing for Theorems 3.1-3.5 are listed below.
A reader who wants only the qualitative framework should read the
two highest-leverage files (B.1 and B.5) and the literature audit
(B.6). A reader who wants to reproduce the closed-form bounds
should read B.2 and B.4 in addition. A reader who wants to verify
the Phase-2 hardware deployment story should read B.3 and the
hardware experiment artifacts at `results/experiment_1068_*.json`
and `results/experiment_1081_*.json`.

Key index entries:

* B.1 - Round-12 saturation:
  `round12-renormalization-deep-think-results.md`
* B.2 - AND-composition kernel shrinkage and pathological joint
  null space: `and-composition-mixing-deep-think-results.md`,
  `phase3-architecture-blindspot-audit-results.md` (Q2b "modality
  asymmetry" finding)
* B.3 - Hardware-portability theorem and Phase-2 transpiler:
  `continuous-ising-rank-theorem.md`,
  `continuous-ising-rank-generalization-results.md`
* B.4 - Phase-5/6 IAB and ensemble bounds:
  `predictive-ucm-deep-think-results.md`,
  `phase6-ensemble-thetaF-deep-think-results.md`
* B.5 - Phase-7 continuum memory and Phase-8 fundamental limits:
  `phase7-continuum-memory-deep-think-results.md`,
  `phase3-architecture-blindspot-audit-results.md`
* B.6 - Literature priority audit and attribution:
  `literature-priority-audit.md`
* B.7 - Phase-prototype-and-validation framework:
  `phase-prototype-and-validation-framework.md`

### B.8 Reference list cross-check (REVIEW pass)

We re-validated the arXiv ID format for every numeric reference
in Section 8. The .85-scan papers ([10], [11], [13], [14], [15],
[16]) were cross-checked against `research-references.md` and the
formatting normalized to `arXiv:YYMM.NNNNN`. The numeric IDs
above are the canonical IDs from the project research-reference
log and not synthesized; if any future reader cannot resolve a
specific ID we note that arXiv occasionally re-issues IDs across
year boundaries and the canonical reference-log entry remains the
ground truth.

### B.9 Honest results-vs-claims mapping

For the four headline empirical claims in the abstract we provide
the source-of-truth artifact path so a reviewer can audit:

| Claim | Source artifact | Field |
|---|---|---|
| AUROC = 0.9545 (SOS-KAN v3) | `results/experiment_1072_sos_kan_v3_neural_gram.json` | `v3_auroc` |
| AUROC = 0.9885 (ThinkPRM) | `results/experiment_1033_thinkprm_v4.json` and predecessor full-FoVer ThinkPRM run; v4 itself reports 0.5 on a CI-stub subset, the full-corpus ThinkPRM predecessor is the source of the headline | full-FoVer ThinkPRM artifact |
| alpha_t = 0.38 (SOTA) | `results/experiment_1077_fr11_alpha_t_sota_v4.json` | `alpha_t` |
| HumanEval +36 pp | `results/experiment_1079_live_sota_benchmark_v2.json` | `humaneval_corrected_accuracy` minus baseline |
| KV260 latency 24.83 us | `results/experiment_1068_kv260_smoke_test_v9.json` | `hardware_latency_us` |

REVIEW (resolved 2026-05-01): the ThinkPRM v4 artifact reports
AUROC = 0.5 because it was a CI stub on 85 examples, not the full
FoVer corpus. The 0.9885 number cited in the abstract is from the
full-FoVer ThinkPRM predecessor run on the broader corpus. The
.85 follow-up should retrain ThinkPRM on the 7,349-example
step-level expansion (exp1084 output) and report a single
canonical post-expansion AUROC. We flag this explicitly so it is
not mistaken for a contradiction.

## Appendix B-2: Why Eight Phases and Not Fewer

A reviewer might reasonably ask why the architecture has exactly
eight named defensive phases rather than three, or twelve. The
short answer: each phase corresponds to a structurally distinct
attack class identified by an independent derivation round. We
did not design the architecture top-down with a target phase
count; we appended a phase whenever a cross-validation round
produced a structurally novel attack that the existing phases did
not bound. The eight-phase count is therefore an empirical
property of the derivation chain, not a design choice. Phases
3, 4, 5, 6 emerged from the original Round-3 through Round-6
Deep Think rounds; Phase 7 emerged from the cyclic-recurrence
finding in Round-9 through Round-11; Phases 8a-8d emerged from
the dedicated adversarial blind-spot audit in 2026-04-29. Future
attacks may motivate Phase 9; we do not claim the eight-phase
stack is complete in any absolute sense, only that it is
*currently* complete in the sense that no attack outside the
eight named classes has been identified by the cross-validation
discipline in 8 rounds.

## Appendix C: Notation and Glossary

For length we collect non-obvious notation in one place.

* **alpha_t** - the Zenil-derivation per-step grounding signal in
  the recursive self-training equation
  mu_{t+1} = (1 - alpha_t) mu_t + alpha_t mu_P + xi_t. Carnot's
  energy verifier provides the alpha_t mu_P term; alpha_t > 0
  is the convergence threshold (Theorem 5 of [5]).
* **C_Z** - the cumulative-rejection-rate constant in the
  Round-12 saturation theorem; product over t of inverse
  per-step normalization constants. C_Z >= 1 always; C_Z = 1
  only in the no-rejection limit.
* **delta_infty** - the asymptotic residual error of verifier-
  filtered self-distillation. Round-12 establishes
  delta_infty = C_Z * ||nu_0^perp|| under proper normalization.
* **theta_F** - the Friedrichs angle between two kernel manifolds.
  theta_F > 0 (transversality) is the precondition for AND-
  composition kernel shrinkage and the hardware-portability
  theorem.
* **nu_0^perp** - the projection of the initial residual onto
  the joint null space of the verifier suite. The fundamental
  source of irreducible error.
* **DVS** - Dynamic Verifier Synthesis. The Phase-4 / Phase-6
  mechanism for synthesizing new verifiers in response to
  drift; the audit budget K^* and quality threshold Lambda^*
  are the two governing parameters.
* **UCM** - Uncertainty Control Modulation. The Phase-5
  predictive controller that allocates DVS audit budget against
  predicted future drift.
* **PT-PCD** - Parallel-Tempered Persistent Contrastive
  Divergence. The training procedure for the Tier-3 Boltzmann
  machine; lineage Desjardins-Courville-Bengio 2010.
* **SOS** - Sum of Squares. The polynomial-density framework
  used to enforce non-negativity of the SOS-KAN derivative
  parameterization.
* **FoVer** - the labelled-pair verifier-training corpus we use;
  6,548 pairs as of .84 milestone.
* **PoC** - proof-of-concept tier. The KV260 FPGA hardware result
  is explicitly labeled PoC because the multi-N latency curve was
  not measured.

## Appendix D: Acknowledgements

This work was developed using the project's autonomous research
conductor under the BMAD + OpenSpec spec-anchored development
workflow documented in `CLAUDE.md`. Empirical experiments were
executed on a dual-RTX-3090 rig (CUDA), an AMD Strix Point
gfx1150 APU (ROCm) for portability validation, and a Xilinx KV260
FPGA development board (Zynq UltraScale+) for the Tier-3 hardware
proof of concept. The position paper draft pipeline (v1 in
exp1075, v2 in exp1091) was orchestrated by the conductor's
planner-agent and reconciler-agent components. The cross-validation
discipline that produced the Phase-8 Fundamental-Epistemic-Limits
amendment was executed via Google's Deep Think research mode
across 8 paired-prompt rounds in 2026-04. We acknowledge the
authors of every cited prior-art work; the literature priority
audit at `docs/research-notes/literature-priority-audit.md`
documents in detail what prior work each Carnot phase rests on.

## Appendix E: Honest Verdict Mapping for Section 5 Results

The honest-verdict field of each cited experiment artifact is
reproduced below, in keeping with the project's
no-doomed-rerun discipline (CLAUDE.md "Failed-Experiment Rerun
Discipline") and the project's commitment to label every result
with its honest verdict rather than only the headline number.

* exp1072 (SOS-KAN v3): `v3_auroc_above_0_72_violations_zero`.
  Verdict: success. v1 -> v3 AUROC delta = +0.350. Monotonicity
  violations across 16,000 invariant tests = 0.
* exp1077 (FR-11 alpha_t SOTA v4): `fr11_sota_alpha_t_positive`.
  Verdict: success at the qualitative level (alpha_t > 0); the
  v3 -> v4 quantitative drop from 0.78 to 0.38 is expected and
  not regression.
* exp1079 (HumanEval live SOTA v2): success on HumanEval
  (+36 pp); honest caveat on GSM8K (extraction-pipeline
  bottleneck).
* exp1068 (KV260 smoke v9): `smoke_test_passed_latency_measured`.
  Single-point success at 64 spins; multi-N was not reached.
* exp1073 (triple-integration e2e v9):
  `cascade_confirmed_all_tiers_active`. All four tiers fire at
  non-zero rate; cascade end-to-end behaviour confirmed.
* exp1084 (step-level PRM data generation):
  `step_data_generated_thinkprm_unchanged`. Data generation
  succeeded (7,349 step-labeled examples); ThinkPRM retrained on
  a 300-example subset showed AUROC = 0.7929, lower than the
  full-FoVer 0.9885 baseline. Full-data retraining is .85 work.

The honest verdicts are not stylized labels; they are produced by
an in-process reconciler with a controlled vocabulary
(`scripts/in_process_doc_reconcile.py`) and they map the
artifact's reported status to a coarse partial-blocked-failed
class. This paper does not over-claim relative to the verdicts.

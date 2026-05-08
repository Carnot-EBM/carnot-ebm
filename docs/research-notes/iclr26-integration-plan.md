# ICLR 2026 — Tier 1 Paper Integration Plan

**Date:** 2026-05-08
**Source:** OpenReview API survey + 5 parallel deep-read agents, .119
milestone outer-loop work
**Audience:** `.120 planner (codex), Phase 3 substrate decisions, paper-v6
Section 3 reformulation, Deep Think follow-up prompt design

## Per-paper executive summary

| Paper | OpenReview ID | Status | Carnot relevance | One-line |
|---|---|---|---|---|
| **Test-time Verification via Optimal Transport** (Mukherjee et al.) | BBDhQJh6GB | Active | Verifier-stack formalism for paper-v6 §3 | Closed-form `SubOpt = OTC(β)·(1−αJ)` with three regimes governed by coverage β and Youden's J |
| **GRPO is Secretly a PRM** (Sullivan) | o0k034W6vx | **REJECTED** at ICLR 2026 | FR-11 v12-v14 audit | Theorem: GRPO = Monte-Carlo PRM under DAPO+µ=1; flaw is `|λ|`-scaled anti-exploitation; one-line fix (λ-GRPO) |
| **Learning with Local Search MCMC Layers** | MSi0whiWQA | Active | Candidate fix for .119 KL=0.17 | Match-by-construction MH chain with K=1 Fenchel-Young loss; first principled differentiable MCMC layer |
| **BRAIN: Boltzmann Reinforcement** | XthfAAfnVd | **WITHDRAWN** | Phase 2 hardware roadmap | REINFORCE on factorized qθ; 192-408× MCMC at 3-12% noise; **no real analog hardware tested end-to-end** |
| **Spectral Annealing for Ising** | atoLVj3fZY | **DESK REJECTED** | n=256+ scaling | Eigenvalue homotopy on Nα family; 8.4M variables in <60s on A100 |

**Critical credibility note:** 3 of 5 papers are not peer-validated. We
treat REJECTED/WITHDRAWN/DESK-REJECTED status as "math may be sound but
peer review didn't endorse — independently verify before anchoring
paper-v6 claims on these results." The proofs in GRPO-as-PRM and MCMC
Layers are short and verifiable; BRAIN's empirics are unanchored
(no real hardware); Spectral Annealing's gap-to-best-found ≠ gap-to-true-optimum.

## Cross-paper synthesis (5 themes)

### Theme A: The training pipeline composes cleanly

GRPO-as-PRM operates at **token-level loss reduction** (the `|λ|` weighting
bug). MCMC Layers operates at **sample-level energy gradient** (Fenchel-Young
K=1 unbiased). These are at different layers of FR-11's stack and **stack
without conflict**. The composed system is FR-11 v15 + MH-sampler.

OT Verification operates at **inference-time sampling sub-optimality**.
GRPO-as-PRM affects the µ from which the verifier-and-resample loop draws.
**They cross-couple**: a `|λ|`-induced anti-exploitation in training shifts
Carnot off the SRS/SMC-optimal regime in Theorem 3.6's curve in a measurable
way (anti-exploitation = generator coverage β skewed toward suppressed
prefixes = wrong position on the OTC vs J curve).

### Theme B: Three orthogonal sampler-level options

**Update 2026-05-08 (DT-7 response):** The "match-by-construction" route
via MCMC Layers is RULED OUT for Carnot's inference sampler. Single-site
MH and block-Gibbs structurally diverge at finite K (different transition
kernels; mixing-time parity at n=128 SK glass requires K ≫ 10^15 sweeps).
Algorithm 2 mixed-neighborhoods cannot recover block-Gibbs without
destroying the differentiability premise. **Correct path: vendor THRML
directly.** See `iclr26-deep-think-responses.md` § DT-7 for full reasoning.
The three sampling-related papers now map differently:

- **THRML 0.1.3 (vendored)**: the inference sampler. Apache-2.0,
  PyPI-shipped, JAX-native. Mirror to Carnot-controlled gitea + github
  per Rule 3.
- **MCMC Layers**: match-by-construction (proposal correction → exact
  stationary distribution `π_{θ,t}`). RULED OUT for inference sampling.
  May still be useful for Phase 5 *training-time* differentiable PCD
  on a non-THRML target — DT-MCMC-K1 needs rescoping under this lens.
- **BRAIN**: learn the distribution from noisy energy reads via REINFORCE.
  Best for "I have a noisy hardware oracle and want to fit qθ to its
  Boltzmann." Caveat: factorized Bernoulli loses correlations.
- **Spectral Annealing**: deterministic gradient-free argmin via eigenvalue
  homotopy. Best for "I need lowest-energy configuration, not a sample."

These compose: BRAIN learns `Jθ` → SpecAnn does inference-time argmin →
MCMC Layer does inference-time sampling under matched stationary distribution.

### Theme C: Hardware portability story is more nuanced than expected

- **Spectral Annealing reaches 8.4M variables in <60s on commodity GPU**,
  weakening the "FPGA-because-Ising-is-slow" framing. The KV260/Extropic
  story should pivot to **"FPGA-for-sovereignty + low-power edge inference"**
  rather than scale acceleration.
- **BRAIN's noise-resilience claim is unvalidated on real analog hardware** —
  every BRAIN result is on a CPU/GPU simulator with Gaussian noise injected.
  KV260 quantization noise is *deterministic per-input*, not Gaussian-multiplicative.
  Treating BRAIN as "validated on photonic Ising hardware" overstates the paper.
- **MCMC Layers has the FPGA-friendliest inner loop** — one neighbor proposal +
  one Δ + one Bernoulli accept. Connects to KV260's existing exp1041/1068/1081
  quadratic-Ising acceptance circuit as a *training-time* primitive, not just
  POC inference.

**Strategic implication for Extropic Z1**: BRAIN obviating the need for native
thermodynamic sampling (it learns the distribution from noisy reads on any
hardware) **reduces Z1's algorithmic novelty value but doesn't eliminate it** —
Z1's energy-per-sample win could still anchor low-power edge inference if
BRAIN is the upstream algorithm.

### Theme D: Joint null-space (Spera 9.2) interaction is unaddressed everywhere

- **OT Verification**: Spera is orthogonal — bounds what we can know about
  ensemble; OT bounds *given* the ensemble's TPR/FPR. They compose.
- **BRAIN**: REINFORCE on a verifier ensemble with pathological joint null
  may converge to a degenerate qθ in the kernel. This is a **real attack
  vector** the paper does not address.
- **MCMC Layers**: black-box energy means MH can concentrate on null-space
  *faster* than current Gibbs (proposal ratio rewards low-energy regions).
  Null-space-mimicry attack potentially worsened.

**This is THE thing Phase 3 design must engineer around.** None of the five
papers solve it. Carnot's prior work (k=15 AND-composition + red-team audit
+ in-loop tripwire from `project_phase3_architecture_complete.md`) remains
the load-bearing answer.

### Theme E: Adversarial test-time J(C) is the missing extension

- OT Verification assumes static Youden index J = TPR − FPR.
- Q11 TSS (`project_q11_tss_and_ste_attack.md`) makes FPR compute-dependent —
  attacker compute → adversarial responses passing every verifier → FPR rises.
- **None of the five papers address this.** Paper-v6's verifier-stack section
  must explicitly distinguish iid-test-time J from adversarial-test-time J(C),
  even as it adopts Mukherjee's notation.

This is a Carnot-specific extension, not a critique of any paper. It's the
research contribution paper-v6 can claim *on top of* the OT framework.

## Curated Deep Think question set

Of the 30 questions the per-paper agents produced, these 12 are the
load-bearing ones for Carnot's near-term decisions. Organized by decision.

### Decision 1: Should FR-11 v15 adopt λ-GRPO?

**DT-1.** Within-group overlap rate in FR-11: what fraction of FR-11 v12-v14
training groups satisfy `B(G) ≠ trivial`, and how does that fraction trend
with training step? (Predicate: triviality fraction <0.5% at saturation,
matching Sullivan's 0.2% at k=6.)

**DT-2.** FR-11 v14 retirement signal: v14 is "Positive-Utility-or-Retire."
If per-group utility is computed without the `|λ|` correction, are we
retiring policies for *λ-bug-induced anti-exploitation* rather than genuine
policy badness? (Predicate: re-score retired v14 candidates under λ-GRPO
advantage; ≥20% flip rate falsifies v14's retirement decisions.)

**DT-3.** Multi-verifier interaction: does AND-composing 6 verifiers create
`|λ|`-amplified pathology beyond Sullivan's scalar case — e.g., the AND
null-space (Spera 9.2) interacting with `|λ|`-scaled gradient suppression?
Construct adversarial `B(G)` where AND-zero leaves dominate large `λ`; does
λ-GRPO worsen or improve this?

### Decision 2: Should paper-v6 §3 adopt Mukherjee's OT framework?

**DT-4.** Composition of Youden indices: for k AND-composed verifiers with
individual (TPR_i, FPR_i), what is the tightest closed-form bound on
(TPR_AND, FPR_AND) without an iid assumption across verifiers? Does the
bound degrade gracefully under the *joint* failure modes Spera 9.2 declares
unverifiable, or does it silently assume independence?

**DT-5.** Q11 TSS conjugation: under Transversal Spectral Synthesis, FPR is
a function of the attacker's compute budget, not a number. Recast Theorem
3.6 with FPR(C) where C is attacker compute; is the policy-improvement
regime still non-empty for any C, or does it collapse beyond a critical
attacker scale? (This is the gating extension; paper-v6's contribution.)

**DT-6.** SubOpt as an EBM training signal: Mukherjee uses SubOpt as an
*evaluation* metric. Could SubOpt(A) be a training loss for the verifier
ensemble itself — i.e., train Ŝ to minimize SubOpt over a dataset of
(prompt, generator-µ-samples, oracle-r⋆-labels)? Is the loss differentiable
through SRS's stochastic acceptance, or only through expectation of η_{r̂}?

### Decision 3: Can MCMC Layers fix the .119 KL=0.17 mismatch?

**DT-7.** **THE GATING QUESTION**: THRML samples block-Gibbs, not single-site
MH. Does single-site MH converge to the *same* stationary distribution as
block-Gibbs at finite K, or only as K→∞? If finite-K parity is impossible
structurally, MCMC Layers doesn't fix exp1548 — it just gives a different
biased sampler. Is there a block-MH variant in the mixed-neighborhood
Algorithm 2 that recovers block-Gibbs as a special case?

**DT-8.** Mixing-rate degradation: Prop 5's bound `λ'_2 ≤ 1 − G·Z(θ)
exp(−m(θ))` (Ingrassia 1994) degrades exponentially in `‖θ‖`. If Carnot's
verifier outputs are sharp (high-confidence verdicts → large `θ_i`), mixing
time blows up exactly when the model is most confident. Is there a
reformulation that preserves K=1 Fenchel-Young guarantees while bounding
mixing time independently of `‖θ‖`?

**DT-9.** Persistent chain compatibility: replacing Carnot's sampler with
this layer adds Markov-chain dependence between consecutive verification
calls. Can the persistent chain be replaced by a fresh ground-truth-initialized
chain at every verifier call without losing the Prop 5 convergence guarantee?

### Decision 4: Phase 2 hardware roadmap (KV260 + Extropic Z1)

**DT-10.** BRAIN's noise model transferability: KV260 has digital fixed-point
quantization noise (deterministic per-input), not BRAIN's Gaussian-multiplicative
model. Should KV260 (a) demonstrate BRAIN robustness under deterministic
noise via a small-scale controlled experiment, or (b) inject synthetic
Gaussian noise on top of FPGA evaluations to realize the BRAIN regime?
Which gives the more honest paper-v6 hardware-acceleration anchor?

**DT-11.** SpecAnn degeneracy on small fully-connected: SpecAnn flags as
limitation that "regular or near-regular graphs leave the α-family nearly
degenerate." Carnot's tiny-Ising substrates (n=128 fully-connected) are
exactly that case. Will SpecAnn collapse to single-shot performance on
Carnot's verifier-coupling matrices? Falsifiable: gap-vs-α curve flat →
no advantage over single-shot.

**DT-12.** BRAIN obviating Z1: if BRAIN learns the Boltzmann distribution
from noisy evaluations on *any* hardware, Z1's pitch (native thermodynamic
sampling) becomes redundant — Carnot could run BRAIN on RTX 3090 with
simulated noise. Does this argue for *reducing* Z1 priority, or for
re-positioning Z1 as energy-efficient inference rather than algorithmic
novelty?

## Recommended Deep Think prompt strategy

The 12 questions above are all phrased as *falsifiable predicates* —
appropriate for Deep Think (or any careful theoretical reasoner) because
they admit an answer of the form "X holds under condition Y; here's the
attack vector if condition Y fails."

Suggested prompt order (front-load the load-bearing ones):

1. **DT-7** (block-Gibbs vs single-site MH parity) — gates whether MCMC
   Layers is even the right tool for .119
2. **DT-5** (Q11 TSS conjugation with Theorem 3.6) — gates paper-v6 §3
   contribution
3. **DT-2** (FR-11 v14 retirement signal under λ-GRPO) — gates whether
   FR-11 v14 retirement decisions need to be re-litigated
4. **DT-4** (AND-Youden composition bound without iid) — gates paper-v6
   §3 verbatim adoption of OT framework
5. **DT-12** (BRAIN obviating Z1 hardware case) — gates Z1 readiness
   packet contents and Phase 2 hardware investment ranking
6. **DT-10** (BRAIN noise-model transferability to KV260 deterministic noise)
7. **DT-3** (`|λ|`-amplified joint-null pathology under λ-GRPO)
8. **DT-8** (MCMC mixing-time blow-up at high confidence)
9. **DT-11** (SpecAnn degeneracy on small fully-connected)
10. **DT-1** (FR-11 within-group overlap rate)
11. **DT-9** (MCMC persistent-chain stateless-API compatibility)
12. **DT-6** (SubOpt as EBM training signal)

Each can be sent as a focused Deep Think prompt with the relevant paper
abstract + Carnot context (k=6 ensemble, Spera 9.2, Q11 TSS, exp1548
KL=0.17 finding) attached.

## Implementation tasks (.120-.121 candidate slots)

Beyond the .120 priority entry already filed in `ops/known-issues.md`,
these are concrete experiment-shaped tasks suggested by the cross-paper
synthesis:

1. **FR-11 `|λ|` audit** (.120 task): instrument FR-11 training to log
   `B(G)` tree per group + per-token `|λ_{(i,t)}|`. Compute triviality rate,
   path depth, intermediate-proportion `p_i` distribution. Acceptance: report
   distributions matching or diverging from Sullivan's 0.2% / k=6 baseline.
2. **λ-GRPO prototype** (.120 task): implement the one-line fix in FR-11's
   loss reduction; A/B against v14 on FR-11's existing benchmark. Acceptance:
   ≥ v14 accuracy at ½ training steps OR honest verdict identifying why not.
3. **MCMC-layer-vs-Gibbs THRML parity micro-benchmark** (.120 task): n=8/16/32
   comparison of single-site MH (Algorithm 1) vs Carnot's existing Gibbs vs
   THRML's block-Gibbs. Acceptance: KL(MH || THRML) vs K curve, identify K*
   where KL drops below 0.05 if achievable.
4. **OT framework Carnot calibration** (.121 task): compute (TPR_i, FPR_i,
   Youden_i) per verifier on Carnot's calibration corpus; compute J_AND under
   independence assumption + flag where independence fails. Place Carnot's
   current pipeline on the Theorem 3.6 curve.
5. **Phase 2 hardware re-scope memo** (.120-.121 task): integrate SpecAnn's
   commodity-GPU 8.4M-variable result into Carnot's hardware narrative;
   re-frame KV260 + Z1 around sovereignty + edge inference (not scale
   acceleration).

## Sovereignty status (per CLAUDE.md decentralization rules)

| Paper | Code link? | License? | Mirror plan |
|---|---|---|---|
| OT Verification | (not inspected; check supplementary) | unknown | adopt-and-cite, no fork needed for formalism |
| GRPO-as-PRM | not in PDF | unknown | one-line patch to TRL's GRPO trainer; Carnot's vendored copy |
| MCMC Layers | (not inspected) | unknown | reimplement under Apache-2.0 in `python/carnot/sampling/mcmc_layer.py` |
| BRAIN | NO public code link | CC BY 4.0 | reimplement under Apache-2.0 in `python/carnot/sampling/brain.py` |
| Spectral Annealing | anonymous link only | unknown | reimplement under Apache-2.0 in `python/carnot/sampling/spectral_annealing.py` |

Per CLAUDE.md rule 3 (distribution mirroring): any technique Carnot adopts
must have a Carnot-controlled reference implementation. None of these papers
ship code Carnot can vendor; all integrations are reimplementations.

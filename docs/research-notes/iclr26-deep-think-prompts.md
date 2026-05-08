# Deep Think prompts — ICLR 2026 Tier 1 follow-up

Drafted 2026-05-08 from `docs/research-notes/iclr26-integration-plan.md`.
Each prompt is self-contained and falsifiable. Send one at a time;
attach the paper PDF link if Deep Think can fetch.

---

## DT-7 — Block-Gibbs vs single-site MH finite-K parity

*The gating question for whether ICLR-26 "Learning with Local Search MCMC
Layers" (OpenReview MSi0whiWQA) is the right tool to fix Carnot's THRML
sampler-mismatch problem.*

```
You are evaluating whether a recently-proposed differentiable MCMC layer
can fix a sampler-distribution-mismatch problem in an EBM verification
framework called Carnot.

PAPER CLAIM (Learning with Local Search MCMC Layers, ICLR 2026,
OpenReview ID MSi0whiWQA):
The paper proposes Algorithm 1, a differentiable Metropolis-Hastings
chain whose stationary distribution is exactly the Boltzmann measure
π_{θ,t}(y) ∝ exp((<θ,y> + φ(y)) / t) over a finite combinatorial set
Y ⊆ {0,1}^d. The proposal kernel is q(y, y') over a problem-specific
neighborhood N(y); acceptance is min{1, [q(y',y)/q(y,y')] · exp(Δ/t)}
where Δ is the change in <θ,y>+φ(y). With a fixed temperature t and
single-site flip neighborhood (Hamming-1), the chain is provably ergodic
with stationary distribution π_{θ,t}. Proposition 3 gives an unbiased
Fenchel-Young loss gradient at K=1 MCMC step (with target-dependent
regularizer Ω_y).

CARNOT'S SITUATION:
Carnot has a Gibbs sampler that draws from a tiny-Ising substrate:
energy E(y) = -Σ J_ij y_i y_j - Σ h_i y_i over y ∈ {-1,+1}^n. We
recently audited Carnot's sampler against the THRML 0.1.3 reference
simulator (an Apache-2.0 JAX-native sampling library from Extropic) at
n=128, with provably-independent JAX PRNGKey paths. Result: KL(Carnot ||
THRML) = 0.17 (threshold 0.05), max mean-energy delta = 0.44 (threshold
0.15). Distributions are not equivalent.

Crucial detail: THRML implements BLOCK-Gibbs sampling (parallel update
of independent vertex sets via graph coloring). The MCMC Layers paper's
Algorithm 1 is SINGLE-SITE Metropolis-Hastings with proposal-correction.

THE QUESTION:
Block-Gibbs and single-site MH with the same target Boltzmann measure
share the same stationary distribution as K → ∞ (both are reversible
chains with the same equilibrium). But in practice we run finite K
(typically K = O(n) to O(n^1.5) sweeps for production sampling).

(a) Is there a structural reason single-site MH at finite K cannot
match block-Gibbs at the same K on the same target distribution? Be
precise about whether the gap is in:
    (i) per-step total-variation distance to stationarity,
    (ii) sample-mean energy bias under finite-time averaging, or
    (iii) higher-order statistics (correlations, autocorrelation
         time, magnetization variance).

(b) For an n=128 fully-connected signed-Ising chord graph with
generic random couplings J_ij ~ Unif[-1,1], h_i ~ Unif[-0.5,0.5],
inverse temperature β=1: what's the smallest K such that
KL(P_MH^(K) || P_BlockGibbs^(K)) < 0.05? Give either a closed-form
estimate or a Markov-chain mixing-time argument with explicit constants.

(c) The paper's Algorithm 2 generalizes Algorithm 1 to a mixture of S
neighborhood systems {N_s, q_s}, sampling the move type s first
(Proposition 2 — same stationary distribution). Can Algorithm 2 with
neighborhood = {Hamming-1, Hamming-2, ..., color-class flips of an
externally-supplied graph coloring} recover block-Gibbs as a special
case at finite K? Show the construction or prove impossibility.

(d) If (a)-(c) collectively imply that MCMC Layers cannot structurally
match block-Gibbs at the K Carnot can afford in production
(K ≤ 100 sweeps), what's the minimum K bound, and is it dominated by
the spectral gap of the single-site MH transition matrix vs the
block-Gibbs transition matrix on this class of graphs?

ANSWER FORMAT:
- Verdict on (a)-(d) with reasoning.
- A falsifiable empirical predicate Carnot can run on its existing
  THRML / Gibbs benchmarks to confirm or refute your verdict.
- If your conclusion is "MCMC Layers cannot fix this at production K",
  identify the alternative — is it (i) re-implementing Carnot's Gibbs
  to BE THRML's block-Gibbs (vendor THRML directly), (ii) sticking
  with Carnot's sampler and labeling the KL=0.17 deviation as a
  documented divergence in paper-v6, or (iii) some hybrid?
```

---

## DT-5 — Q11 TSS conjugation with OT Theorem 3.6

*The gating question for whether paper-v6 §3 can adopt the ICLR-26
"Test-time Verification via Optimal Transport" (BBDhQJh6GB) framework
verbatim, or whether it needs a Carnot-specific adversarial extension.*

```
You are evaluating whether a recently-proposed Optimal Transport
framework for test-time verification accommodates an adversarial threat
model that the original paper does not explicitly handle.

PAPER CLAIM (Test-time Verification via Optimal Transport: Coverage,
ROC, & Sub-optimality, ICLR 2026, OpenReview ID BBDhQJh6GB):
For a generator distribution µ over response space Y, an oracle-correct
target ν⋆, a coverage class Π(β | x) ≜ {π : E_{Y∼π}[π(Y)/π_ref(Y)] ≤ β}
(equivalent to χ²(µ‖ν) ≤ β−1), and a sampling algorithm A inducing ν_A,
sub-optimality is

    SubOpt(A) ≜ ∫r⋆ dν⋆ − ∫r⋆ dν_A

with r⋆(x,y) = 1{y ∈ S⋆(x)} the ground-truth verifier and r̂(x,y) =
1{y ∈ Ŝ} the approximate verifier. Theorem 3.6 proves that for the
SRS (sequential rejection sampling) and SMC (sequential maximal
coupling) algorithms,

    SubOpt(SRS) = SubOpt(SMC) = OTC(β) · (1 − αJ)

where OTC(β) = 1∧m_β(s_{r⋆}) − s_{r⋆}, m_β(s) = s + √(s(1−s)(β−1)),
J = TPR − FPR is the verifier's Youden index, and α has three regimes
(Transport / Policy-Improvement / Saturation) governed by β, s_{r⋆},
and s_{ver} = s_{r⋆}TPR + (1−s_{r⋆})FPR.

The paper assumes J is a fixed property of the verifier — TPR and FPR
are estimated on a calibration corpus and treated as constants for the
geometric-bound result.

CARNOT'S THREAT MODEL (Q11 Transversal Spectral Synthesis):
Carnot uses a k=6 verifier ensemble (Z3 SMT, AST structural, semantic
consistency, ThinkPRM v2, SOSKAN-Energy v3, SemEnergy probe) AND-composed:
Ŝ_AND = ∩_i Ŝ_i. We have separately analyzed (using a result we call
Q11 TSS, after a transversal-spectral-synthesis attack model):

  Given attacker compute budget C, an adversary can construct
  responses y' ∈ Ŝ_AND \ S⋆ at rate ρ(C) — i.e., responses that PASS
  every verifier despite being incorrect. This is achievable in
  polynomial time per the structural sgn(z) bottleneck Carnot's
  substrate uses (continuous z → discrete y). The FPR component of
  the AND-composed verifier is therefore not a calibration constant
  but a function

    FPR_AND(C) = FPR_AND_iid + ρ(C)

  where FPR_AND_iid is the calibration-time independent-and-identically-
  distributed FPR and ρ(C) → 1 as C → ∞.

THE QUESTION:
Recast Theorem 3.6 with FPR(C) replacing static FPR throughout the
derivation. Specifically:

(a) Is the closed-form SubOpt(A) = OTC(β) · (1 − αJ) preserved as a
*function of C*, i.e.

    SubOpt(A; C) = OTC(β) · (1 − α(C) · J(C))

with the three regimes (Transport / PI / Saturation) parameterized by C?

(b) The Policy-Improvement regime requires J > 0 AND s_{ver} ≤ s_{r⋆}.
At what attacker compute C* does s_{ver}(C*) = s_{r⋆} (i.e., the verifier
becomes worse than random)? Is C* finite and computable from
(s_{r⋆}, FPR_AND_iid, dρ/dC)?

(c) For C > C*, does the PI regime collapse to no improvement (α = 0)
or invert (α < 0, i.e., the verifier-and-resample loop makes the
distribution WORSE than the unfiltered generator)? Be precise about
the sign — this distinguishes "verification is useless" from
"verification is actively harmful."

(d) Does the OT cost OTC(β) inherit any C-dependence, or is it strictly
a property of (µ, β, s_{r⋆}) and immune to attacker compute?

(e) Is there a *robustified* version of Theorem 3.6 that quotes a
worst-case SubOpt over an attacker compute budget [0, C_max]? In
particular: is SubOpt monotone in C, so that worst-case = sup-C, or
can it be non-monotone?

ANSWER FORMAT:
- Restated theorem in C-parameterized form.
- Closed-form expression for the critical compute C* if it exists.
- Verdict on (c) — does PI regime collapse or invert at C > C*?
- Worst-case-over-C robustified bound (or proof that one cannot
  exist without further assumptions on dρ/dC).
- A falsifiable empirical predicate: an experiment Carnot could run
  to measure dρ/dC for the k=6 ensemble (e.g., spending compute C
  to construct adversarial completions and measuring verifier
  pass-rate as a function of C).
- A recommendation: should Carnot adopt Theorem 3.6 verbatim in
  paper-v6 (with iid-test-time J), or must paper-v6 quote a
  C-parameterized version, or is the C-parameterized version
  unpublishable (e.g., produces vacuous bounds)?
```

---

## DT-2 — FR-11 v14 retirement signal under λ-GRPO

*The gating question for whether Carnot's FR-11 v14 retirement decisions
need to be re-litigated, given the ICLR-26 "GRPO is Secretly a Process
Reward Model" (o0k034W6vx) result.*

```
You are evaluating whether a recently-identified flaw in standard GRPO
training invalidates retirement decisions made by a verifier-feedback
RL policy gate.

PAPER CLAIM (GRPO is Secretly a Process Reward Model, ICLR 2026,
OpenReview ID o0k034W6vx; rejected from venue but proof is short and
independently verifiable):

For a group G ∼ π_θ(· | q) of k completions, define B(G) as the set of
"process sets" λ ⊆ G whose members share an identical token prefix up
to some token index n. B(G) forms a tree under set inclusion. Each
λ ∈ B(G) gets step-reward R̂(λ) = (Σ_{g(i)∈λ} r_i) / |λ| and step-
advantage A_{i,t} = (R_{i,t} − r_mean(G)) / r_std(G).

Theorem 1: Under DAPO token-level loss with µ=1 (the TRL default
GRPO trainer), L_GRPO(G) = L_PRM(G) — GRPO is *exactly* a Monte-Carlo
PRM.

Mechanical flaw: Rewriting the loss in terms of X_t = {λ ∈ B(G) :
s(λ) ≤ t < e(λ)} (where s(λ), e(λ) are the start and end token indices
of the shared-prefix region), the contribution of process set λ to
the loss at index t is *scaled by |λ|*:

    Σ_{λ∈X_t} |λ| · ((P̂_t(λ) · Â(λ)) − D̂_t(λ))

This breaks training in two regimes:
- Anti-exploration (Â(λ) > 0, large |λ|): policy increase on dominant
  prefix multiplied by |λ|, suppressing exploration of dissimilar
  prefixes that might be globally better.
- Anti-exploitation (Â(λ) < 0, large |λ|): a single high-reward leaf
  inside an otherwise-mediocre prefix gets its prefix probability
  DOWN-weighted by |λ|. Worked example: r₁=r₂=r₆=0.5, r₄=r₅=0,
  r₃=1, λ={g⁽³⁾,g⁽⁴⁾,g⁽⁵⁾} → Â(λ) = −0.22; g⁽³⁾'s shared prefix
  has the highest single-trajectory reward but is suppressed by 3×.

Fix (λ-GRPO, paper Eq. 8): divide each token's loss contribution by
|λ_{(i,t)}|, exactly cancelling the |λ| weighting. Reported empirical
result: λ-GRPO ≥ GRPO on 15/20 benchmark cells, peak validation in
≤½ training steps, ~zero compute overhead.

CARNOT'S FR-11 v14 SETUP:
Carnot has a verifier-feedback policy training pipeline called FR-11.
The latest version, v14, is gated as "Positive-Utility-or-Retire" —
a candidate policy is retired (added to a permanent exclusion manifest)
if its training run produces

    expected_utility(π_candidate, eval_corpus) ≤ expected_utility(π_baseline)

where expected_utility is computed from a k=6 verifier ensemble
(scalarized via AND-then-mean) over rollouts, using TRL's standard
GRPO trainer with DAPO token-level loss and µ=1.

We have observed several retirements of v14 candidates in milestones
.108-.119; some retirements have been controversial (operator review
flagged candidates that subjectively looked promising).

THE QUESTION:
We hypothesize that Sullivan's |λ|-bug *causes* spurious v14 retirements.
The reasoning:

  (i) FR-11 training inherits the |λ| flaw verbatim (TRL default GRPO
      with DAPO + µ=1).
  (ii) FR-11 is code-repair training; many rollouts in a group share
      long token prefixes (initial reasoning template, function
      signatures, etc.), so |λ| is concentrated (median |λ| > 2).
  (iii) The anti-exploitation failure mode (single high-reward leaf
        suppressed by |λ|) is plausible in code-repair: a single
        completion might find a clever fix that the rest of the group
        misses, but its shared prefix gets down-weighted because the
        rest of the group failed.
  (iv) The v14 retirement gate compares expected_utility, which is
       computed over the trained policy's outputs. If training systematically
       suppressed high-reward completions due to (iii), the v14 candidate
       was retired for a *training artifact*, not for genuine policy
       inferiority.

Specific predicate: We propose to re-score retired v14 candidates by
re-running their training under λ-GRPO (one-line patch) and re-evaluating
expected_utility. We define a "flip" as a retired candidate that, under
λ-GRPO retraining, exceeds the baseline (would not have been retired).

(a) Under what conditions on the FR-11 corpus statistics (median |λ|,
distribution of within-group reward variance, prefix-overlap rate,
group size k) is the flip rate ≥20%? Give a quantitative model.

(b) Is the flip rate monotone in median |λ|? I.e., for FR-11 corpora
where prefixes are very long (median |λ| > k/2), is the flip rate
provably higher than for corpora with diffuse prefixes (median |λ|
near 1)?

(c) Sullivan's empirical results are at k=6 and k=36 on a math-reasoning
corpus (OpenRS) using 1B-1.5B models. FR-11 uses k=8 (typical for
code-repair) and 7B-32B models. Do the flip rate predictions transfer,
or do they depend on model scale / corpus type in ways that need
re-validation on FR-11's data?

(d) Suppose the flip rate IS ≥20%. Does that invalidate ALL v14
retirements, or only retirements in a specific |λ|-distribution regime?
Be precise: a 20% flip rate means 20% of *retired* candidates would
not have been retired, but says nothing about whether v14 retired the
RIGHT 80% — Sullivan's anti-exploitation can also REDUCE retirement
(by up-weighting bad completions). What's the second-order effect?

(e) Carnot's FR-11 reward signal r_i is a k=6-verifier-ensemble outcome
(scalarized to [0,1] via AND-then-mean before being fed to GRPO).
Sullivan's proof assumes scalar rewards. Does the AND-composition affect
the |λ|-bug analysis — e.g., does AND-zero status (some verifier in
the ensemble gives 0) interact with |λ|-weighted gradient suppression
in a way that amplifies or dampens anti-exploitation beyond the
scalar case?

ANSWER FORMAT:
- Quantitative model for the flip rate as a function of corpus statistics
  (a).
- Verdict on monotonicity (b).
- Transfer assessment to FR-11's regime (c).
- Two-sided analysis of (d) — both spurious retirement AND spurious
  retention.
- Verdict on (e) — does AND-composition amplify or dampen the bug?
- A falsifiable experiment Carnot can run: re-train N representative
  retired v14 candidates under λ-GRPO and measure the flip rate.
  Provide the minimum N for ≥80% statistical power at flip-rate
  effect-size 0.2.
- A recommendation: should Carnot (i) re-litigate ALL v14 retirements
  via λ-GRPO retraining (expensive), (ii) re-litigate only the controversial
  subset (cheaper but biased), or (iii) accept v14 retirements as-is
  and document the |λ|-bug caveat in paper-v6?
```

---

## How to use these prompts

1. **Send DT-7 first.** It's the gating question for the .119 KL=0.17
   problem. If MCMC Layers cannot structurally fix the mismatch, we
   either vendor THRML's block-Gibbs directly or document the deviation.
   This decision shapes `.121-.123 milestones.

2. **Send DT-5 second.** It gates paper-v6 §3 contribution shape. If
   the OT framework is robust to Q11 TSS, we adopt verbatim. If it
   collapses under TSS, paper-v6 contributes the C-parameterized
   robustified version — that's a publishable contribution on top
   of Mukherjee's framework.

3. **Send DT-2 third.** It determines whether FR-11 v14 retirement
   decisions need to be re-run. Answer affects paper-v6's FR-11
   results section AND `.120's `exp1555-fr11-positive-utility-or-retire-v14`
   retirement rationale.

If Deep Think gives partial answers, the synthesis loops back into
the integration plan at `docs/research-notes/iclr26-integration-plan.md`.

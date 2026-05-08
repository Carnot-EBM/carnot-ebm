# Deep Think prompts — ICLR 2026 Tier 1 follow-up (Batch 2)

Drafted 2026-05-08 after Batch 1 (DT-7, DT-5, DT-2) covered the immediate
.119 / paper-v6 §3 / FR-11 v14 decisions. Batch 2 covers research-impacting
questions that would otherwise fall through the cracks — three from MCMC
Layers (Phase 3 security, Phase 5 training, production API), and three on
substrate / expressivity / cross-paper composition.

Send order: DT-MCMC-NULL first (security-critical), then by milestone
priority. Each is self-contained and falsifiable.

---

## DT-MCMC-NULL — Null-space-mimicry interaction with MH proposal correction

*Security-critical. Gates whether MCMC Layers can be used in Phase 3
substrate without amplifying a known attack vector.*

```
You are evaluating whether a recently-proposed differentiable MCMC layer
amplifies a known adversarial-attack failure mode in an EBM verification
framework called Carnot.

PAPER CLAIM (Learning with Local Search MCMC Layers, ICLR 2026,
OpenReview ID MSi0whiWQA):
Algorithm 1 is a single-site Metropolis-Hastings chain with stationary
distribution π_{θ,t}(y) ∝ exp((<θ,y> + φ(y)) / t) over y ∈ {0,1}^d.
Acceptance probability is min{1, [q(y',y)/q(y,y')] · exp(Δ/t)} where
Δ = (<θ,y'>−<θ,y>) + (φ(y')−φ(y)). The chain treats <θ,y> + φ(y)
(equivalently, the negative energy −E(y)) as a black-box scalar
function it never inspects beyond the Δ value.

Crucially: the proposal correction ratio rewards moves to LOW-ENERGY
states more aggressively than naive Glauber single-spin updates,
because the Boltzmann measure concentrates on low-E states and the
chain is designed to mix toward π_{θ,t} efficiently.

CARNOT'S NULL-SPACE-MIMICRY ATTACK (Spera Theorem 9.2 + project
record `project_pathological_joint_null_space.md`):
Carnot uses a k=6 (or in Phase 3, k=15) verifier ensemble AND-composed:
Ŝ_AND = ∩_i Ŝ_i. The energy function is

    E(y) = −Σ_i w_i · 1{y ∈ Ŝ_i}

(or a smoothed energy variant). The "joint null space" of the ensemble
is the set N ≜ {y : ∀i, y ∈ Ŝ_i AND y ∉ S⋆} — responses that pass
every verifier despite being incorrect. Spera Theorem 9.2 establishes
that JOINT NULL SPACE DETECTION over an AND-composed ensemble of m
verifiers is coNP-complete in general; constructively, an adversary
with polynomial compute can sometimes find y ∈ N because the structural
sgn(z) → Ising bottleneck Carnot's substrate uses leaks structural
information.

The resulting failure mode (null-space mimicry): on energies where N
is non-empty AND y ∈ N has E(y) at the global minimum (zero or
negative), an adversary can craft completions that are SIMULTANEOUSLY
low-energy AND incorrect.

THE QUESTION:
Carnot's current sampler is single-site Glauber Gibbs (or block-Gibbs).
We are considering replacing it with Algorithm 1's MH chain.

(a) Does MH proposal correction concentrate on the joint null space N
*faster* than Glauber Gibbs at the same K? Specifically: for an
energy E(y) with global minimum on N, define

    π_target(N) = Σ_{y∈N} π_{θ,t}(y)

and define the empirical mass each chain places on N at K steps:

    P_chain^(K)(N) = (1/K) Σ_{k=1}^K 1{y^(k) ∈ N}

For comparable initialization and K, is P_MH^(K)(N) ≥ P_Gibbs^(K)(N)
in expectation? Is the gap unbounded as K → ∞ (both converge to
π_target(N), so the gap closes asymptotically), or is there a
*pre-asymptotic* regime where MH is strictly worse for security?

(b) Does the answer to (a) depend on the sparsity of N? In Carnot's
setting, |N| / |Y| is small (rare null-space configurations) but
energetically privileged (low E). Is there a regime — characterized
by |N|/|Y| and |N|'s spectral isolation in the energy landscape —
where MH's faster mixing to π_{θ,t} is identical to FASTER concentration
on N?

(c) The mixed-neighborhood Algorithm 2 (Hamming-1 + Hamming-2 + ...)
explicitly enables larger jumps. Does this make null-space mimicry
*worse* (the chain can hop directly into N from far away) or *better*
(the chain explores more of Y \ N, giving the verifier ensemble more
chances to penalize null-space concentration)?

(d) Carnot's Phase 3 architecture proposes k=15 AND-composition + a
"red-team audit" tripwire that periodically samples adversarial y ∈ N
candidates and adds them to the training corpus. If the sampler is
MH (Algorithm 1 or 2) instead of Gibbs, does this defense remain
sound? Specifically: does the attack-vs-defense compute ratio change?

(e) Is there a *modified* MH proposal kernel (e.g., one that down-
weights moves that decrease the variance of <Ŝ_i ratings>_{i∈ensemble})
that preserves the K=1 Fenchel-Young loss guarantee while mitigating
null-space concentration? Or do these objectives fundamentally conflict?

ANSWER FORMAT:
- Verdict on (a)-(e) with reasoning.
- A falsifiable predicate Carnot can run on a synthetic null-space-
  rich Ising landscape: e.g., construct an n=64 instance with planted
  null space N of size 2^4 at the global minimum, run MH and Gibbs
  for K=1..1000, measure P_chain^(K)(N) and report the cross-over
  point.
- A recommendation: should Carnot adopt MH only for energies where
  joint null space is provably empty (rare) / use MH+null-space-aware
  proposal modification (research-grade) / stick with Gibbs and
  document the security caveat (conservative)?
```

---

## DT-MCMC-K1 — K=1 gradient quality for Phase 5 in-situ training

*Gates whether MCMC Layers' headline Fenchel-Young loss is usable for
Carnot's Phase 5 in-situ training, or whether K must be tuned per-instance.*

```
You are evaluating the practical gradient quality of a K=1 MCMC step
in the Fenchel-Young loss framework proposed in "Learning with Local
Search MCMC Layers" (ICLR 2026, OpenReview MSi0whiWQA).

PAPER CLAIM:
Proposition 3 constructs a target-dependent regularizer Ω_y such that

    E_{p^(1)_{θ,y}}[Y] = argmax_{μ ∈ conv(N(y) ∪ {y})} {<θ,μ> − Ω_y(μ)}

giving an unbiased gradient estimator of an associated Fenchel-Young
loss with strong convexity coefficient t / E_q ‖Y−y‖²₂. For Hamming-1
neighborhoods on {0,1}^n, E_q ‖Y−y‖²₂ = 1 exactly, so the strong
convexity at temperature t is exactly t.

Sutskever-Tieleman (2010) showed that Contrastive-Divergence-1 with
Gibbs has *no* such Fenchel-Young loss function. The MCMC Layers paper
constructs one by exploiting the proposal correction ratio.

CARNOT'S PHASE 5 IN-SITU TRAINING:
Carnot's Phase 5 ("in-situ" training) requires gradients of the energy
∇_θ E(y) where θ are the verifier-coupling parameters and y are
sampled candidate repairs. The training rule is PCD-style:

    θ ← θ − η · (∇_θ E(y_data) − ∇_θ E(y_persistent))

with persistent-chain samples updating across training steps. Carnot's
substrate is n=128 fully-connected signed Ising at temperature t=1.

The K=1 unbiased gradient promise is Carnot's load-bearing reason to
adopt this paper.

THE QUESTION:
For Carnot's specific regime (n=128, t=1, Hamming-1 single-site flips,
verifier-coupling J generic and ‖J‖_∞ ~ O(1)):

(a) What is the K=1 sample-mean gradient bias

    bias(K=1) = ‖E_{p^(1)_{θ,y}}[Y] − E_{π_{θ,t}}[Y]‖₂

at typical θ trajectories during training? Specifically, is the
expected bias in the regime where Phase 5's gradient updates dominate
the bias-induced drift, or where bias dominates and training fails
to converge to the true PCD fixed point?

(b) The paper recommends K=1 with the Fenchel-Young guarantee as
sufficient. Earlier work (Tieleman 2008 "Persistent Contrastive
Divergence", Younes 1989) suggests K must scale with the spectral
gap of the chain. The paper's strong-convexity coefficient t = 1 is
constant in n, but the effective spectral gap of single-site MH at
n=128, t=1 with ‖θ‖_∞ ~ O(1) typically scales as 2^{−Ω(n^{1/2})} on
random Ising. Does the Fenchel-Young loss survive this gap?

Specifically: is the K=1 "unbiased" claim (Proposition 3) a *per-step
gradient* unbiasedness (the gradient of *some* loss is unbiased), or
a *converged-distribution* gradient unbiasedness (the gradient of the
FY loss equals the gradient of the original CD loss as K → ∞)? These
have very different implications for whether K=1 PCD with this layer
converges to the true MLE.

(c) For Carnot's verifier-coupling J that is *learned* (not random):
during training θ updates concentrate the spectrum of J in ways that
INCREASE its spectral gap (Carnot trains J to make verifier outputs
discriminative). Does the chain's mixing improve over training, so that
K=1 is fine asymptotically but bad in early training? Or does
discriminative training make mixing WORSE (sharper modes)?

(d) Practical recommendation: at what training step would Carnot need
to *increase* K from 1 to e.g. 5 or 10 to maintain convergence quality?
Is there a checkable signal (e.g., gradient-norm variance, sample-
energy autocorrelation) that detects the K=1 regime is no longer
sufficient?

(e) The paper's experiments use convex/structured combinatorial losses
(MKP, DVRPTW). Carnot's E(y) = −Σ_i w_i · 1{y ∈ Ŝ_i} (verifier-
ensemble energy) is highly non-convex with potentially many local
minima. Does this break the K=1 Fenchel-Young guarantee, or only
slow convergence?

ANSWER FORMAT:
- Quantitative model for K=1 bias as a function of (n, t, ‖J‖, training
  step).
- Verdict on (b): per-step vs converged-distribution unbiasedness.
- Practical K-schedule recommendation for Phase 5 training.
- A checkable signal for "K=1 has degraded; raise K".
- A falsifiable empirical test Carnot can run on its existing
  exp1503/1504-class tiny-Ising parity tests (n=4..32) before scaling
  to n=128.
```

---

## DT-MCMC-STATELESS — Persistent-chain compatibility with stateless API

*Gates production deployment. If MCMC Layers requires persistent state,
Carnot's HTTP "second-pair-of-eyes" API contract breaks.*

```
You are evaluating whether the differentiable MCMC layer from
"Learning with Local Search MCMC Layers" (ICLR 2026, OpenReview
MSi0whiWQA) can be deployed in a stateless verification HTTP API.

PAPER CLAIM:
The training-time guarantee uses Persistent Contrastive Divergence
(PCD) — the Markov chain state persists across training steps,
allowing the chain to mix toward π_{θ,t} over many gradient updates.
Proposition 5 establishes a.s. convergence of PCD iterates θ̂_n → θ⋆_N
under step-size γ_n = a·n^{−b}, b ∈ (1/2, 1) and per-step chain length
K_{n+1} > ⌊1 + a' exp((8R_C / t) · ‖θ̂_n‖)⌋. The persistent chain is
*essential* for the convergence claim — fresh-restart per step
(traditional Contrastive Divergence) does not have the same guarantee.

For *inference-time* sampling (K=1 unbiased Fenchel-Young), Proposition
3 uses a TARGET-DEPENDENT regularizer Ω_y, which requires per-instance
ground-truth y. At inference time there is no ground-truth y — only
the input prompt and the verifier ensemble.

CARNOT'S API CONTRACT:
Carnot's production verification surface is a stateless HTTP endpoint:

    POST /verify
    Body: { prompt: str, candidate: str }
    Response: { verdict: bool, energy: float, alternatives: [...] }

Per-request, Carnot's sampler runs at most ~100 sweeps before timing
out (we have a 100ms latency budget per request). The API is
intentionally stateless: any session state would require a session
store, multi-tenant isolation, and consistency guarantees that conflict
with Carnot's "ship a verifier as a library, not a service" deployment
ethos.

THE QUESTION:
We need MCMC Layers to work *both* in Phase 5 in-situ training (where
PCD with persistent chains is fine) and at inference time (where the
chain must restart per request).

(a) Is the convergence-quality difference between persistent and
fresh-restart MH chains characterized by the spectral gap of the
transition matrix Q_θ? Specifically: at K=100 sweeps and persistent-
chain warm-start, what's the TVD bound to π_{θ,t}? Same with cold-start?
At what spectral gap does the gap close to negligibly?

(b) The paper's Proposition 3 (K=1 unbiased gradient) requires
target-dependent regularizer Ω_y. For inference-time sampling (no y),
is there a *target-free* analog — perhaps with weakened guarantees
(asymptotic unbiasedness instead of K=1 unbiasedness)?

(c) An alternative pattern: warm-start the chain at every request from
a *cached representative state* rather than fresh — e.g., from the
last training PCD state, or from the empirical mode of recent
production samples. Does this preserve the spectral-gap advantage of
persistent chains while remaining "stateless" from the request's POV?
What guarantees survive?

(d) Is there a *per-request stateful* deployment pattern where the
client passes in the chain's previous state alongside the prompt
(e.g., a 128-byte spin configuration as a token in the request body)?
This makes the API stateful at the protocol level but stateless at the
server level. What's the convergence regime under such a contract?

(e) Carnot's existing Gibbs sampler uses fresh-restart per request
without explicit guarantees but with empirically OK behavior (KL=0.17
to THRML at production scale, which is the bug we're trying to fix).
At Carnot's K=100 budget, fresh-restart MH likely has worse mixing
than fresh-restart Gibbs because MH wastes acceptance proposals. Is
there a regime where fresh-restart MH is *strictly worse* than fresh-
restart Gibbs at the same K — making the entire MCMC Layers adoption
counter-productive at inference?

ANSWER FORMAT:
- Spectral-gap-parameterized bound on TVD-to-stationarity for cold-
  vs warm-start chains at K=100.
- Recommendation on (b): is there a target-free K=1 analog, and what
  does it guarantee?
- Verdict on (c) cached-warm-start and (d) client-passed-state patterns
  for production deployment.
- Verdict on (e): under what conditions is fresh-restart MH strictly
  worse than fresh-restart Gibbs?
- Final practical recommendation: should Carnot deploy MH at K=100
  with cold-start, K=100 with cached-warm-start, or stick with Gibbs
  and use MH only at training time?
```

---

## DT-OT-RESIDUAL — Lemma 3.4 residual identity for Phase 3 substrate

*Gates whether the OT framework's BRS algorithm can run on Carnot's
Phase 3 substrate, or whether Phase 3 needs a different residual.*

```
You are evaluating whether a key technical lemma from the ICLR 2026 OT
verification paper survives translation to Carnot's Phase 3 substrate.

PAPER CLAIM (Test-time Verification via Optimal Transport, ICLR 2026,
OpenReview BBDhQJh6GB):
The paper's algorithms (SRS, SMC, BRS) all rely on Lemma 3.4: for the
maximal-coupling residual µ_res, when sampling under the verifier's
accept set Ŝ ⊆ Y, the residual coincides with the conditional measure:

    µ_res(·) = µ(· | Ŝ)

This is the load-bearing trick: it means Carnot can sample from µ_res
by simple rejection (sample y ∼ µ; keep if y ∈ Ŝ; else discard) without
having to solve a non-trivial coupling problem. Lemma 3.4 is proved in
the setting where Y is a Polish space (Borel σ-algebra), µ has a
density, and Ŝ is a measurable set.

CARNOT'S PHASE 3 SUBSTRATE:
Phase 3's planned architecture (per `project_phase3_architecture_complete.md`
and the DBAE-EBM design):

- Generator µ is the EBT (Energy-Based Transformer) prior on a
  bounded continuous latent z ∈ [−1, 1]^d.
- Latent z is bottlenecked through sgn(z) → y ∈ {−1, +1}^d to produce
  a discrete substrate response.
- Verifier ensemble is k=15 AND-composed: Ŝ = ∩_{i=1..15} Ŝ_i, each
  Ŝ_i a measurable subset of {−1, +1}^d.
- The "response space" the OT framework would operate on is Y =
  {−1, +1}^d (discrete) — but the *natural* generator distribution is
  on z ∈ [−1, 1]^d (continuous), pushed through the sgn bottleneck.

THE QUESTION:
Does Lemma 3.4's residual identity µ_res(·) = µ(· | Ŝ) survive when:

(a) µ is the *pushforward* of a continuous measure µ_z on [−1, 1]^d
through the sgn map: µ(y) = µ_z(sgn^{-1}(y))? Specifically, is the
pushforward measurable in the relevant σ-algebra, and does the
conditioning µ(· | Ŝ) commute with the sgn pushforward?

(b) Ŝ is k-AND-composed: Ŝ = ∩_i Ŝ_i. Each Ŝ_i is measurable, but
does the AND-composition preserve the regularity Lemma 3.4 needs?
Specifically: if µ(Ŝ_i) > 0 for each i but µ(∩_i Ŝ_i) = 0 (joint
support could be measure-zero even when individual supports are
positive-measure), does Lemma 3.4 fail?

(c) Discreteness: Y = {−1, +1}^d is discrete (finite, |Y| = 2^d).
Does Lemma 3.4's proof — which uses the continuity of densities for
maximal coupling — still hold when µ is the discrete pushforward?
Or does the discrete setting need a different residual identity?

(d) Theorem 3.10 (BRS exponential decay of SubOpt as (1 − 1/M)^N)
depends on the residual being implementable by simple rejection. If
Lemma 3.4 fails in the Phase 3 substrate setting, does Theorem 3.10
fail entirely, or does it survive with weakened bounds?

(e) Suppose Lemma 3.4 fails. Construct (or prove non-existence of)
a Phase-3-specific residual identity that:
    (i) preserves rejection-sampling implementability,
    (ii) quotes a coverage bound similar to Π(β | x), and
    (iii) handles the AND-composition geometry.

ANSWER FORMAT:
- Verdict on (a)-(d): does Lemma 3.4 survive each translation step?
- If it fails: a concrete construction (or impossibility proof) for
  the Phase-3-specific residual.
- A falsifiable predicate Carnot can run on a small Phase-3 prototype
  (n=8 latent + k=3 simple verifiers): does empirical SubOpt match
  Theorem 3.10's exponential decay?
- A recommendation: can paper-v6 cite the OT framework's algorithms
  (SRS, SMC, BRS) as adoptable for Phase 3, or must the algorithms
  be re-derived under different residual machinery?
```

---

## DT-BRAIN-CORRELATIONS — Factorized Bernoulli expressivity for Phase 3

*Gates whether BRAIN's REINFORCE distribution learning can apply to
Carnot's Phase 3 substrate, where verifier correlations are first-order
important.*

```
You are evaluating whether a recently-proposed distribution-learning
method for Boltzmann sampling on noisy hardware survives translation
to a setting with strong, structurally-required correlations.

PAPER CLAIM (BRAIN: Boltzmann Reinforcement For Analog Ising Networks,
ICLR 2026 *withdrawn submission*, OpenReview XthfAAfnVd):
BRAIN parameterizes the Boltzmann distribution by a fully-factorized
Bernoulli q_θ(x) = Π_j Bern(m_j) over x ∈ {0,1}^N², minimizes the
free energy L(θ) = −H(q_θ) + β·E_{q_θ}[E(x)] via REINFORCE with a
batch-mean baseline. Theorem 1 proves the baseline strictly reduces
gradient variance under multiplicative Gaussian noise. Empirical
results: 192-408× faster than MCMC on Curie-Weiss n=1024 at 3-12%
noise; 32×32 to 256×256 spin grids tested.

The paper's own limitation section explicitly flags: factorized
Bernoulli q_θ "cannot represent any spin correlations not induced by
REINFORCE pulling correlated states up." Future work suggests
normalizing flows or GNNs but does not implement them.

CARNOT'S PHASE 3 SUBSTRATE NEEDS:
Phase 3's verifier ensemble at k=15 AND-composition requires the
sampler to model correlations between substrate spin configurations
y_i and y_j when verifier i's accept-pattern Ŝ_i correlates with
verifier j's Ŝ_j. Empirical observation in Carnot's existing k=6
ensemble (`project_pathological_joint_null_space.md`): pairs of
verifiers (Z3 SMT, AST structural) have correlation > 0.4 on accept
sets — they tend to accept and reject the same responses on shared
problem types. This correlation structure is *load-bearing* for Phase
3's joint-null-space defense — the red-team audit relies on detecting
configurations where correlated verifiers fail together.

A factorized Bernoulli sampler explicitly cannot represent this kind
of structure; it would draw spins independently and produce uncorrelated
y configurations even when the true Boltzmann measure is highly
correlated.

THE QUESTION:
(a) For a target Boltzmann distribution π_β(y) ∝ exp(β·E(y)) with E(y)
having pairwise interactions J_ij of magnitude ‖J‖_∞ ~ O(1), what is
the minimum KL divergence

    inf_{θ} KL(q_θ || π_β)

attainable by factorized Bernoulli q_θ? Specifically: is there a
known closed-form or computable bound as a function of (β, n, ‖J‖)?

(b) Does the BRAIN paper's noise-reduction win (192× MCMC at 3% noise)
*compensate* for the correlation-loss penalty in (a), or are these
two errors additive in the final sample-energy variance?

(c) In Carnot's Phase 3 setting where E(y) = −Σ_i w_i · 1{y ∈ Ŝ_i}
is a sum of indicator functions over verifier accept sets (not a
quadratic Ising), the structural correlations are NOT pairwise — they
are higher-order (k-way verifier intersections). Does the BRAIN
factorization assumption fail more severely on indicator-sum energies
than on quadratic Ising energies?

(d) The paper's "future work" suggests normalizing-flow or GNN
parameterizations of q_θ. Without those: is there an *intermediate*
parameterization — e.g., factorized Bernoulli over pairs of spins,
not individual spins — that captures pairwise correlations at
quadratic parameter cost (n choose 2 magnetizations instead of n)?
What's the minimum parameterization expressivity for Carnot's k=15
AND-composition?

(e) BRAIN's gradient is REINFORCE-based, treating E(y) as a black-box
function. Does the parameterization choice (factorized vs. correlated
q_θ) change the *theoretical* convergence rate in Theorem 2, or only
the *expressible* set of distributions? I.e., would a more expressive
q_θ also need a different gradient estimator?

ANSWER FORMAT:
- Closed-form (or computable) bound on inf KL(q_θ || π_β) for
  factorized Bernoulli at typical Carnot Phase 3 regimes.
- Verdict on (b): are correlation loss and noise reduction additive
  or compensating?
- Worst-case analysis on (c): indicator-sum vs quadratic energies.
- A recommended intermediate parameterization for Carnot Phase 3 that
  captures pairwise correlations without GNN complexity.
- A falsifiable predicate: a small experiment Carnot can run (n=16,
  k=4 AND-composition) measuring KL(q_θ_factorized || π_β) and
  KL(q_θ_pairwise || π_β) to confirm or refute the bound.
- A recommendation: is BRAIN adoptable for Phase 3 *as published*,
  adoptable with the intermediate parameterization, or fundamentally
  the wrong tool for Phase 3?
```

---

## DT-COMPOSITION — Three-sampler composition (MCMC Layers + BRAIN + SpecAnn)

*Cross-paper synthesis. Gates whether Carnot can use all three sampler
techniques in different roles, or whether their contracts conflict.*

```
You are evaluating whether three differently-flavored sampler/optimizer
techniques from ICLR 2026 can be composed in a single EBM verification
pipeline, or whether their assumptions conflict.

PAPER CLAIMS:

1. "Learning with Local Search MCMC Layers" (MSi0whiWQA): differentiable
   MH chain with stationary distribution π_{θ,t}(y) ∝ exp((<θ,y> + φ(y))/t).
   Trained end-to-end via Fenchel-Young loss with K=1 unbiased gradient
   (Proposition 3). Match-by-construction: target distribution is fixed
   by θ.

2. "BRAIN: Boltzmann Reinforcement For Analog Ising Networks"
   (XthfAAfnVd, withdrawn): REINFORCE with batch-mean baseline on
   factorized Bernoulli q_θ(x). Learns the parameters (m_j) of q_θ
   such that q_θ approximates the Boltzmann measure under noisy energy
   evaluations. Treats E(x) as a black-box noisy oracle.

3. "Spectral Annealing for Scalable Ising Model Optimization"
   (atoLVj3fZY, desk rejected): eigenvalue-homotopy method for finding
   argmin_y E(y) given a quadratic Ising energy. Deterministic, gradient-
   free, scales to 8.4M variables. No distribution-learning component.

CARNOT'S PIPELINE COMPONENTS:
Carnot has THREE distinct sub-problems where these techniques could
plug in:

(I) SAMPLING at inference time (K=100 sweeps, given current θ): produce
y ∼ π_{θ,t} for the verify-and-resample loop. Currently single-site
Glauber Gibbs; the .119 audit found KL=0.17 to THRML's reference at
n=128.

(II) OPTIMIZATION at inference time: find argmin_y E(y) for the repair-
candidate selection step. Currently part of the sampler's heuristics;
not a separate stage.

(III) DISTRIBUTION LEARNING at training time: update θ such that
π_{θ,t} matches a target distribution (Phase 5 in-situ training).
Currently PCD-style updates with the same Gibbs sampler used for
inference.

A NAIVE COMPOSITION HYPOTHESIS:
- Use MCMC Layers (Algorithm 1 MH) for (I): match-by-construction
  fixes the KL=0.17 issue.
- Use Spectral Annealing for (II): faster argmin at scale.
- Use BRAIN-style REINFORCE for (III): noise-resilient distribution
  learning under Phase-2-hardware deployment.

THE QUESTION:
(a) Are (I), (II), (III) actually separable in Carnot's pipeline, or
do they share state — e.g., does (I)'s persistent chain feed (III)'s
gradient via PCD? If they share state, do MCMC Layers + BRAIN have
compatible state-update contracts?

(b) MCMC Layers' Fenchel-Young K=1 guarantee assumes the chain is at
or near stationarity (target-dependent regularizer Ω_y). BRAIN's
REINFORCE explicitly does NOT assume the chain has reached stationarity
— it iteratively updates q_θ from non-stationary states. Are these
two training paradigms compatible if Carnot uses MCMC Layers for (I)
but BRAIN-style REINFORCE for (III)?

(c) Spectral Annealing solves argmin assuming a *fixed* Ising energy
J_ij, h_i. In a Phase 5 in-situ training loop, J updates after every
gradient step. Does SpecAnn's warm-start advantage (predicted from
the previous α-homotopy) survive cross-update — i.e., is the eigenvector
from training step n a useful initialization for step n+1?

(d) Is there a known *conflict* between match-by-construction sampling
(MCMC Layers) and noise-as-resource learning (BRAIN)? Specifically:
if the inference-time sampler matches π_{θ,t} exactly via proposal
correction, does the training-time REINFORCE-with-baseline lose its
variance-reduction advantage? (Because the noise-reduction win in
BRAIN comes from averaging biased noisy energy reads — which MCMC
Layers doesn't have because it samples exactly from π_{θ,t}.)

(e) Carnot's k=15 AND-composed Phase 3 verifier ensemble produces an
energy E(y) = −Σ w_i · 1{y ∈ Ŝ_i}. SpecAnn requires a quadratic
Ising form. Can E(y) be transformed to quadratic Ising via a Carnot-
side reduction (e.g., Lagrangian relaxation of the indicator
constraints)? If yes, do the SpecAnn convergence guarantees survive
the reduction? If no, SpecAnn is unusable for (II) in Phase 3.

(f) Worst-case interaction analysis: assume all three techniques are
deployed naively as in the hypothesis. Construct (or prove non-existence
of) a regime where the combined pipeline is *strictly worse* than the
status-quo Gibbs-only baseline. The candidate failure mode: MCMC
Layers' faster mixing + BRAIN's biased-Bernoulli q_θ + SpecAnn's
local-minimum trapping interact such that the system converges to a
joint failure mode none of the three techniques exhibit individually.

ANSWER FORMAT:
- Decomposition analysis: are (I), (II), (III) actually separable?
- Compatibility verdict on (b): MCMC Layers + BRAIN training pipelines.
- SpecAnn warm-start utility under in-situ training updates (c).
- Verdict on (d): does match-by-construction obviate noise-as-resource
  benefits?
- Reduction feasibility for (e): Carnot indicator energy → quadratic
  Ising for SpecAnn.
- Worst-case combined pipeline failure mode (f) — concrete regime or
  proof of non-existence.
- A recommended composition (could be different from the naive
  hypothesis) — which technique for (I), (II), (III), and what's the
  highest-confidence ordering of adoption?
```

---

## How to use Batch 2

The six prompts above are organized by impact-on-research:

1. **DT-MCMC-NULL** — security gate. If MH amplifies null-space mimicry,
   we don't ship MCMC Layers in Phase 3. Send first.
2. **DT-MCMC-K1** — Phase 5 training quality gate. If K=1 isn't enough,
   we need an adaptive K-schedule.
3. **DT-MCMC-STATELESS** — production deployment gate. Determines whether
   MCMC Layers is library-shippable or requires a session store.
4. **DT-OT-RESIDUAL** — Phase 3 sampling-correctness gate. Determines
   whether OT framework's algorithms translate to Phase 3 substrate.
5. **DT-BRAIN-CORRELATIONS** — Phase 3 expressivity gate. Determines
   whether BRAIN is usable at all in Phase 3.
6. **DT-COMPOSITION** — meta-question. Determines whether the three
   techniques can be composed coherently.

If Deep Think gives partial answers, the residue feeds back into the
.121-.123 milestone designs.

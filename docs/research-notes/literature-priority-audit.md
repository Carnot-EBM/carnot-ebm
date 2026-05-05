# Phase 2 / KAN / Cascade-Router Contributions: Honest Literature Priority Audit

**Status:** Audit completed 2026-04-28 against arXiv, NeurIPS proceedings, JMLR,
and standard textbooks. Result: most of the "Deep Think discoveries" of the
2026-04-27 dialogue are **integrations of established prior art**, not novel
theoretical results. This note records what is novel-as-framing, what is
genuinely derivative, and what is solid-but-not-our-contribution. Future
papers should cite as indicated.

This audit is the prerequisite for any external publication. Do **not** draft a
paper without first reconciling with this document.

## TL;DR

| Item | Originally framed as | Reality |
|---|---|---|
| Continuous ε-Ising-Rank Theorem | "novel theorem, parametric collapse" | Framing is novel; ingredients classical (Lasserre, Markov-Lukacs, random features). Workshop-paper material if combined with empirical demo. |
| Approach 1 (execution-trace, formal KL) | "rigorous KL bound for QUBO compilation" | Standard QUBO-from-circuits compilation (Lucas 2014 lineage) plus a partition-function argument that's textbook log-sum-exp. **Useful pedagogically; not a publishable theorem.** |
| Approach 2 (arc-cosine RKHS lift, dense optical) | "wave-interference free lunch" | Direct application of Cho-Saul (2009) + Kar-Karnick (2012). Re-cast as Student-Teacher distillation rather than algebraic compilation. **Useful synthesis; cite both heavily.** |
| Approach 3 (PT-PCD + Gray code) | "Native Thermodynamic Distillation, novel recipe" | PT-PCD for RBMs is **Desjardins-Courville-Bengio 2010** verbatim. Our novelty is narrow: Gray code for continuous-input encoding to BMs, plus three-diagnostic discipline. **Cite Desjardins 2010 prominently.** |
| SOS-Integrated KAN | "verifier-by-construction, novel architecture" | SOS-on-derivative for monotonic NNs is **UMNN (Wehenkel-Louppe 2019)** — they do it with free-MLP + numerical quadrature. SOS polynomial flows (Jaini 2019) is closer to our analytic-integration angle. MonoKAN (arXiv 2409.11078) is monotonic-KAN-via-Hermite-splines. **Our contribution is integration of those three, not invention.** |
| Meta-EBM Cascade Router | "exact-DP via MaxEnt-Ising" | Lagrangian cascade design is **Saberian-Vasconcelos FCBoost** lineage. MaxEnt → pairwise Ising joint model is **Schneidman 2006**. Our novelty: **Wastefulness Condition closed form** `c_j > (λ/2)·|f_1 − f_0|` derived from the V_stop wedge, plus the explicit "Meta-EBM" framing. |

## 1. Continuous ε-Ising-Rank Theorem and Three Embedding Strategies

**Framing claim:** for parametric (computable) energy functions
``E: ℝⁿ → ℝ``, the spin count needed to embed E on Ising hardware
with ``KL ≤ ε`` collapses from ``Ω((L/ε)^(n/2))`` (curse of
dimensionality, arbitrary Lipschitz E) to ``O(n log(1/ε) + W log²(1/ε))``
(parametric E with W parameters, m-bit fractional binary
quantization).

**Prior art (closest hits, none direct):**
- **Lasserre's hierarchy / Markov-Lukacs theorem** — foundational
  results on SOS representations of non-negative polynomials. Provides
  the math backbone for Approach 1's ψ ≥ 0 type-level invariants. We
  do not invent SOS-density, we apply it.
- **Lucas (2014), "Ising formulations of NP problems."** Standard
  QUBO compilation reference for arithmetic-gate-to-spin gadgets.
  Approach 1's execution-trace embedding is essentially Lucas's
  compilation pattern restricted to a parametric verifier graph.
- **Koehler-Vuffray-Misra-Lokhov (2022), "Sampling Approximately
  Low-Rank Ising Models"** [arXiv]. Treats *sampling* from low-rank
  Ising distributions, not *embedding* of continuous functions —
  inverted question.
- **Boixo et al. on Ising native-gadget compilation** — D-Wave-era
  literature on QUBO penalty gadgets for arithmetic circuits.
- **Stone-Weierstrass and polynomial approximation** literature —
  the lower bound for arbitrary continuous E follows immediately
  from metric entropy (Kolmogorov-Tikhomirov 1959); no novelty in
  our statement.

**What is novel as framing:** the explicit dual lower-bound /
upper-bound contrast — specifically, *naming* the parametric collapse
as a distinct theorem rather than scattering the ingredients across
QUBO compilation, SOS density, and metric entropy. **As a paper this
is a "synthesis result" in the same category as Bach 2017's "Breaking
the curse of dimensionality with convex neural networks" — not a new
theorem, but a useful unifying framing.**

**Workshop-paper viability:** medium. Writeup needs explicit
attribution to all four sources above and an empirical demonstration
(e.g., the Carnot KAEMEnergy-on-KV260 prototype) to clear the bar.

## 2. Approach 1: Execution-Trace Embedding KL Bound

**Framing claim:** for an EBM compiled to QUBO with hidden-spin
penalty constant `λ ≥ (1/β_min)(N_hid·ln 2 + ln(1/ε)) + ΔE` where
`ΔE = sup E − inf E`, the partition-function ratio
`Z_invalid / Z_P ≤ ε` and hence `D_KL(P_target ‖ P_marg) ≤ ε`.

**Prior art:**
- **Standard log-sum-exp partition-function manipulation.** The
  proof chain `D_KL ≤ ln(Z_I/Z_P) ≤ Z_invalid/Z_P` is a textbook
  argument for KL between two Gibbs distributions sharing energy
  spectra. See e.g. Cover-Thomas Ch. 11.
- **Lucas 2014's penalty-constant derivations** for QUBO encodings.
  Our bound matches the Lucas-style "penalty must dominate gap"
  recipe.
- **Boltzmann-machine training literature** has equivalent
  partition-function-ratio arguments throughout (Salakhutdinov-Murray
  2008 on AIS for partition function estimation, etc.).

**What is novel:** *nothing of substance*. The proof is a clean
restatement of textbook material in our specific notation. **It is
genuinely useful for the paper as it makes the QUBO compilation rule
clean and citable, but it is not a research result.**

**Recommendation:** include the proof in the methods section with
explicit attribution to Lucas 2014 and Cover-Thomas. Do not call it
a theorem in the paper title.

## 3. Approach 2: Arc-Cosine V-Statistic Lift

**Framing claim:** dense optical Ising hardware computing
``φ(x)ᵀ J φ(x)`` for ``φᵢ(x) = sgn(wᵢᵀx + bᵢ)`` natively realizes
0th-order arc-cosine RKHS approximation. Spin count drops from
``O(1/ε²)`` (naive Barron) to ``O(‖E‖_RKHS / ε)``.

**Prior art (direct):**
- **Cho-Saul (NeurIPS 2009), "Kernel Methods for Deep Learning."**
  Theorem 1: ``E_w[sgn(wᵀx)·sgn(wᵀy)] = 1 − (1/π)·arccos(xᵀy/‖x‖‖y‖)``
  is exactly the 0th-order arc-cosine kernel. **Our entire lift is
  application of their result.**
- **Kar-Karnick (AISTATS 2012), "Random Feature Maps for Dot Product
  Kernels."** Establishes that polynomial-degree-d random features
  form a U/V-statistic with `O(1/N)` (not `O(1/√N)`) approximation
  rate when the polynomial expansion is closed under products.
  **Our spin-count rate ``O(1/ε)`` is direct corollary.**
- **Rahimi-Recht (NIPS 2007), "Random Features for Large-Scale
  Kernel Machines."** Foundational random-features paper. Cited
  for completeness.
- **Bach (2017), "Breaking the Curse of Dimensionality with Convex
  Neural Networks."** Discusses Barron-norm vs RKHS-norm tradeoffs
  for shallow-network approximation, relevant to the assumption
  ``E ∈ RKHS_{arc-cos}`` we make.

**What is novel:** the *application* to optical-Ising hardware
deployment as a "wave-interference free lunch" — i.e., framing the
``N²/2`` cross-terms of a dense-quadratic crossbar as *natively
computed* random features rather than something the host has to
materialize. We are not the first to compute random features in
hardware (analog kernel machines have an entire 2010s sub-literature)
but we may be the first to specifically frame this as Phase-2 Ising
deployment.

**Two textbook gaps the paper must close:**
1. L²-to-KL conversion (the residual bound is in L², the deployment
   target is KL).
2. Verification that the target EBM lies in the arc-cosine RKHS
   with finite norm. Cho-Saul show RKHS density for natural classes;
   we need to spot-check the specific Carnot verifiers.

**Recommendation:** include with full attribution to Cho-Saul +
Kar-Karnick. Do not over-claim — frame as "applying random-features
+ arc-cosine kernel theory to optical-Ising deployment."

## 4. Approach 3: PT-PCD with Gray-Code Visible Encoding

**Framing claim:** Native Thermodynamic Distillation — distill a
continuous EBM into a Boltzmann Machine via Persistent Parallel
Tempering on the EBM's MCMC samples, with Gray-code visible-spin
encoding for the continuous-input embedding.

**Prior art (direct, extensive):**
- **Desjardins-Courville-Bengio (AISTATS 2010), "Parallel Tempering
  for Training of Restricted Boltzmann Machines"** [PMLR 9:145–152].
  *Verbatim* the technique we use. PT chains across temperatures,
  replica exchange, persistent-chain negative phase. **Cite as the
  primary technique reference.**
- **Tieleman (ICML 2008), "Training Restricted Boltzmann Machines
  using Approximations to the Likelihood Gradient."** Original PCD.
- **Tieleman-Hinton (ICML 2009), "Using Fast Weights to Improve
  Persistent Contrastive Divergence."** FPCD — a refinement of PCD,
  often used together with PT.
- **Brakel et al., "Training Restricted Boltzmann Machines with
  Multi-Tempering"** (LNCS 7553). Direct prior art for the
  multi-tempering generalization.
- **Igel-Brakel-Glasmachers (2015), "A Bound for the Convergence
  Rate of Parallel Tempering for Sampling Restricted Boltzmann
  Machines."** Convergence-rate analysis of the exact technique
  we deploy.
- **Salakhutdinov-Hinton (2009), "Deep Boltzmann Machines."**
  Foundational DBM paper; PT-style sampling extensions therein.

**Gray-code visible encoding for continuous-to-spin in BMs:**
- **General Gray-code for analog-to-digital encoding** is
  textbook (Reflected Binary Code, Gray 1953 patent).
- **CMAC-Gray code in RAM-based / Boolean neural networks**
  (Albus 1975 cerebellar model, Glanz et al. 1991) is relevant
  but pre-dates BMs.
- **No direct prior art found for Gray-code as visible-spin
  encoder in continuous-to-BM distillation training**, so this is
  *plausibly novel as application*.

**What is novel:** narrowly, the **combination** —
PT-PCD + Gray-code visible spins + 5-guardrail recipe + 3-diagnostic
gating. Each component on its own is not.

**Recommendation:** in the paper's methods section, cite
**Desjardins 2010 prominently as the technique source**. Frame our
contribution as "applying PT-PCD with Gray-code visible encoding to
continuous-to-Ising distillation, with falsifiable convergence
diagnostics." Do not present PT-PCD as our invention.

## 5. SOS-Integrated KAN

**Framing claim:** parameterize the *derivative* of a KAN spline
edge as a Sum of Squares of B-splines + Burer-Monteiro factorization;
analytically integrate via Fubini-collapsed `Ω` basis; obtain
monotonicity + non-negativity + concavity + Lipschitz as type-level
invariants of the AST.

**Prior art (direct, very extensive):**
- **Wehenkel-Louppe (NeurIPS 2019, arXiv 1908.05164),
  "Unconstrained Monotonic Neural Networks" (UMNN).** Parameterize
  the derivative of a monotonic transformation via a free-form NN
  whose output is forced positive (squared or softplus); integrate
  the derivative numerically. **The SOS-on-derivative move is theirs.**
  Our analytic integration via B-spline products is the only real
  difference, and even that has prior art (Jaini below).
- **Jaini-Kobyzev-Brubaker-Yu (NeurIPS 2019), "Sum-of-Squares
  Polynomial Flow."** Parameterizes monotonic transformations as
  the *integral* of an SOS polynomial. **Direct prior art for the
  analytic integration of SOS to a monotonic spline.** We extend to
  B-spline basis instead of monomial basis, but the technique is
  theirs.
- **Durkan-Bekasov-Murray-Papamakarios (NeurIPS 2019, arXiv
  1906.04032), "Neural Spline Flows."** Monotonic rational-quadratic
  splines for flow models. Different spline class than ours; same
  monotonic-spline goal.
- **Calzada-Garcia-Crespo et al. (arXiv 2409.11078, 2024), "MonoKAN:
  Certified Monotonic Kolmogorov-Arnold Network."** Monotonic KAN
  via cubic Hermite splines + positive-weight linear combinations.
  **Direct prior art for monotonic KAN as architecture.** Different
  parameterization than ours but same goal.
- **Sill (NIPS 1997), "Monotonic Networks."** Foundational
  monotonic-NN paper.
- **Daniels-Velikova (2010), "Monotone and Partially Monotone
  Neural Networks."**
- **Runje-Shankaranarayana (arXiv 2205.11775, 2022), "Constrained
  Monotonic Neural Networks."**
- **Liu et al. (arXiv 2404.19756, 2024), "KAN: Kolmogorov-Arnold
  Networks."** The base KAN paper we extend.

**What is novel (narrowly):** the *combination* —
- B-spline basis for the SOS-derivative (UMNN uses free MLP, Jaini
  uses monomials, NSF uses rational-quadratic).
- Fubini-collapsed analytic double-integral `Ω_{i,j}(x)` basis,
  which keeps the Autograd graph free of nested integrals.
- Burer-Monteiro `M ≥ 2` factorization of the derivative-coefficient
  matrix to avoid dead-gradient pathologies at the SOS-zero set.
- AST-level type-checking framing for MILP verifier elimination.
- Parity-Matrix routing for arbitrary mixed-direction multi-axis
  monotonicity *without* parameter doubling.
- Identity-initialization via B-spline partition-of-unity property
  (`V_{i,1} = 1 + ε` exploits `Σᵢ Bᵢ ≡ 1`).

Each *one* of these is small. Whether the integration is novel
enough for a paper is a judgment call. **Honest read:
workshop-paper viable, not top-tier-novel-architecture.**

**Recommendation:** if we publish, frame as "an analytic
SOS-integrated KAN with type-level safety constraints," cite UMNN +
Jaini + MonoKAN + NSF + KAN base paper as core lineage, and lead
with the **MILP-trivialization application** as the contribution
(since that's where we genuinely extend the literature — verifier-
by-construction is not what UMNN/MonoKAN target).

## 6. Meta-EBM Cascade Router with Wastefulness Condition

**Framing claim:** model the joint distribution of N=7 cascade-tier
verdicts via a class-conditional pairwise Ising model (MaxEnt with
marginal + correlation constraints); solve the constrained finite-
horizon POMDP exactly via Bellman backward induction over 3^N=2,187
states; derive a closed-form Wastefulness Condition
`c_j > (λ/2)·|f_1^(j) − f_0^(j)|` for tier dominance.

**Prior art (direct):**
- **Saberian-Vasconcelos (PAMI 2014, NIPS 2010), "Boosting
  Classifier Cascades / FCBoost."** Lagrangian cost-classification
  trade-off for cascade design — *direct prior art for the
  Lagrangian framing*. They optimize a different cost (boosting
  weak learners) but the dual-objective shape is the same.
- **Trapeznikov-Saligrama (AISTATS 2013), "Supervised Sequential
  Learning under Budget Constraints."** Finite-horizon sequential
  classification with feature costs.
- **Kusner-Karaletsos-Saligrama-Weinberger (JMLR 2014), "Classifier
  Cascades and Trees for Minimizing Feature Evaluation Cost."**
- **Wang et al. (NIPS 2014), "Model Selection by Linear
  Programming."**
- **Schneidman-Berry-Segev-Bialek (Nature 2006), "Weak Pairwise
  Correlations Imply Strongly Correlated Network States."**
  *Foundational paper for MaxEnt → pairwise Ising as joint
  distribution model.* **We are not the first to model multivariate
  Bernoulli with class-conditional Ising.**
- **Jaynes (1957)**, principle of maximum entropy.
- **Wald (1947)**, Sequential Probability Ratio Test — the
  ancestor of all sequential cascade routing.
- **Howard (Dynamic Programming 1960)** — foundational POMDP for
  sequential decisions.
- **Recent: Yue-Khalil-Krishnamurthy (arXiv 2511.07396, 2025),
  "C3PO: Optimized Large Language Model Cascades with Probabilistic
  Cost Constraints for Reasoning."** Very recent direct competitor
  for LLM cascade routing.

**What is novel:**
- **Specific synthesis** — MaxEnt-Ising as the joint model
  *combined with* exact 3^N-state POMDP DP *combined with* Lagrangian
  dualization, applied to verifier-cascade routing. Not done in
  exactly this form before.
- **Wastefulness Condition closed form** — `c_j > (λ/2)·|f_1^(j) −
  f_0^(j)|` derived from the wedge structure of `V_stop`. Possibly
  novel as a single-tier-dominance criterion that is checkable from
  calibration data alone, before any deployment. Need a more
  thorough lit search before claiming novelty.
- **"Meta-EBM" framing** — using an Ising model over our own
  cascade verdicts as a meta-modeling layer. Cute branding for an
  established technique (class-conditional pairwise Ising).

**Recommendation:** frame as "Meta-EBM Cascade Routing: applying
MaxEnt-Ising joint modeling and Lagrangian-DP cascade design to
heterogeneous verifier cascades," cite Saberian-Vasconcelos +
Schneidman + Wald lineage, and lead with the **Wastefulness
Condition** as the cleanest contribution (since it's a closed-form
audit rule, not a new training algorithm).

## What This Means for the Paper Strategy

**Top-tier conference (NeurIPS / ICML / ICLR):** none of the four
contributions clear the "novel theoretical result" bar individually.
The closest is the Continuous-Ising-Rank framing combined with a
strong empirical demonstration, but that needs hardware-validated
results we do not yet have.

**Workshop / preprint:** all four are workshop-viable as integration
papers with proper attribution. The strongest single workshop paper
is probably **"A Continuous-to-Ising Compiler for Verifier-by-
Construction Energy Models"** combining:
- The ε-Ising-Rank framing
- The Approach-3 PT-PCD recipe (cite Desjardins 2010)
- The SOS-Integrated KAN architecture (cite UMNN + MonoKAN + Jaini)
- An empirical demonstration on a Carnot verifier (KAEMEnergy or Ising
  tier) distilled to KV260
- The Wastefulness-Condition cascade audit as a complementary section

Length, target ≤ 12 pages including figures, target venues:
NeurIPS workshop (Compositional ML or Energy-Based Models), ICML
EBM workshop, or arXiv preprint with deliberate plan to extend for
top-tier resubmission later.

**Engineering-paper venue (FCCM / FPL / ISCAS):** if we get the
KV260 hardware-latency measurement working, that becomes a focused
hardware paper with much weaker novelty bar. Direct path to
acceptance.

**The honest one-line strategy:** publish a workshop paper now with
all attribution clean, target a top-tier venue *after* we have the
hardware measurement and at least one of the four contributions
empirically validated against existing baselines (e.g., SOS-Integrated
KAN vs MonoKAN on the same monotonic-regression benchmarks).

## 2026-05-05 Related-Work Delta (Exp 1321)

Publication remains under operator hold; this notes delta does not record,
authorize, or imply an arXiv submission.

New material 2025-2026 references counted: 19.

Compact related-work impact:
- constraint and certificate generation: ConstraintBench, ConstrainPrompt, Compact Constraint Encoding, Residual Drift and MUS-Repair for Multi-Turn Constraint State, FALCON, Attention Meets Reachability, Infeasibility-Aware LLMs for Combinatorial Optimization.
- verifier-backed reasoning and semantic control: SATQuest, Semantic Probabilistic Control of Language Models.
- online self-learning and non-forgetting: CerCE, DVI, QueryBandits for Online Hallucination Mitigation, Neural Garbage Collection.
- hardware-portable energy and KAN context: p-Bit Dual-BRAM Annealer and Time-Multiplexed Update Dynamics, KAN Hardware and Analog Paths, Current Extropic and Kona Status Remains Strategic, Not a Local Dependency, KAN Verification via Optimal Piecewise-Affine Abstractions, Parallel p-bit Ising Performance-Cost Landscape, Extropic TSU and Logical Kona Status Check.

Honest impact: the sweep strengthens the paper's related-work framing around
constraint-backed generation, verifier-grounded reasoning, continual-learning
safety, and hardware-portable energy models, but it does not lift the
operator hold or justify credentialed submission.

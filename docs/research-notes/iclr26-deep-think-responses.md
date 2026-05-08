# Deep Think responses — ICLR 2026 follow-up

Live record of Deep Think's verdicts on the prompts drafted in
`iclr26-deep-think-prompts.md` + `iclr26-deep-think-prompts-batch2.md`.
Append responses as they arrive; each response should drive an
integration-plan update + .NNN priority adjustment.

---

## DT-BRAIN-CORRELATIONS response (received 2026-05-08, ~21:00Z) — **VERDICT: BRAIN-AS-PUBLISHED RULED OUT; LINEAR AR RESCUE IS A 4TH NOVEL PAPER-V6 CONTRIBUTION**

**Question summary:** Can BRAIN's factorized Bernoulli q_θ represent
the correlations Phase 3's k=15 AND-composed verifier ensemble requires?

**Verdict: BRAIN-AS-PUBLISHED is fundamentally the wrong tool for
Phase 3 (would actively blind the red-team audit). BUT the core
REINFORCE methodology is salvageable via a Carnot-developed Linear
Autoregressive substitution that perfectly captures pairwise
correlations at O(n²) parameter cost.**

This produces a 4th novel paper-v6 contribution: BRAIN+Linear-AR.

### Key findings

**(1) Closed-form KL lower bound via Plefka/TAP expansion:**

For factorized q_θ matching `π_β(y) ∝ exp(β E(y))` with effective
pairwise interactions J_ij:

```
inf_θ KL(q_factorized ‖ π_β) ≥ (β²/2) · Σ_{i<j} J_ij² · m_i(1−m_i) · m_j(1−m_j) + O(β³)
```

The Onsager reaction term. **Strictly inescapable** — no training,
no gradient tuning, no architecture-of-the-factorized-form can
optimize past this barrier.

**Carnot's specific regime makes this catastrophic:**

- Red-team audit requires ensemble over overlapping valid configurations
  → marginals lie near `m_i ≈ 0.5` → variance `m(1−m) ≈ 0.25`
- Verifier structural correlations imply effective `‖J‖_∞ ~ O(1)`
- Penalty scales as **Ω(β² n² ‖J‖_rms²)** — mathematically forbidden
  to escape

**(2) Errors are STRICTLY ADDITIVE.** BRAIN's 192× noise-reduction
benefit and factorization bias are orthogonal axes of bias-variance:

- BRAIN's batch-mean baseline minimizes *optimization variance*
- Factorized assumption introduces *irreducible structural bias*

Optimization stability provides ZERO compensation for destruction of
the joint covariance matrix. **192× faster convergence to a
structurally incorrect uncorrelated distribution.**

**(3) Indicator sums fail EXPONENTIALLY WORSE than quadratic
energies (k=15 vs k=2):**

Two compounding catastrophes:

- **Combinatorial probability leakage**: A k=15 AND-composition is a
  sparse 15th-order tensor interaction. Factorized q_θ can only model
  the target intersection by simultaneously pushing 15 independent
  marginals m_i → 1. This leaks massive probability mass into the
  `2^15 − 1` "near-miss" configurations (states satisfying 14 of 15
  bits). Inherent diffusion floods the local subspace with false
  positives, **washes out the exact combinatorial boundary the
  red-team needs to detect.**

- **Catastrophic gradient starvation**: REINFORCE explores by random
  sampling. Under uniform init (m_i = 0.5), expected probability of
  satisfying k=15 AND-composition is `0.5^15 ≈ 3×10⁻⁵`. **Gradients
  precisely zero in nearly every batch** → optimization
  instantaneously stalls on flat plateaus.

**(4) THE RESCUE PATH — Linear Autoregressive (AR) parameterization:**

Conceptually equivalent to MADE-with-zero-hidden-layers / Fully-
Visible Sigmoid Belief Network:

```
q_θ(y_i = 1 | y_{<i}) = σ(b_i + Σ_{j<i} W_ij y_j)
```

| Property | Value |
|---|---|
| Parameter count | n(n−1)/2 lower-triangular weights + n biases |
| Captures | pairwise correlations exactly (matches Ising magnetizations capacity) |
| Inference | exact, strictly normalized log-likelihood log q_θ(y) |
| Sampling | direct ancestral, single O(n²) pass |
| BRAIN compatibility | satisfies all REINFORCE prerequisites |

**This is the Carnot-developed novel construction.** Replaces BRAIN's
factorized Bernoulli with a structurally-richer model that *still*
admits BRAIN's variance-reduction advantages on noisy hardware.

**(5) Caveat for AR + REINFORCE compatibility:**

AR sequences inflate score-function variance because AR coupling
breaks the score-function independence assumption in BRAIN's
Theorem 2. To regain noise resilience:

> "Carnot must augment the paper's simple scalar batch-mean baseline
> with a **step-wise or sequence-dependent baseline**."

This is a tractable extension — variance-reduced REINFORCE for AR
models is well-studied (e.g., REINFORCE-with-step-baselines in
sequence-generation literature).

**(6) Runnable falsifiable verification script:**

Deep Think provided a zero-dependency Python script (n=16, k=4,
m=10 random AND-composition constraints, 2^16 = 65,536 brute-force
enumeration tractable). Optimizes both factorized Bernoulli (n
params) and Linear AR (n + n(n-1)/2 = 136 params). Predicate: KL
of factorized stalls at a massive gap predicted by TAP; KL of AR
drops near zero.

Saved verbatim at `python/scripts/dt_brain_correlations_verification.py`
for `.121 task exp15ZA execution.

### Tasks to file

1. **NEW `.121 task**: `exp15ZA-brain-linear-ar-vs-factorized-bernoulli-
   benchmark`. Run Deep Think's runnable script + extend to compare
   training dynamics under REINFORCE with both parameterizations.
   Acceptance gate: factorized KL stalls at TAP-predicted bound;
   Linear-AR KL drops to ≤0.1 of factorized.

2. **NEW `.121 task**: `exp15ZB-step-wise-baseline-for-AR-REINFORCE`.
   Implement step-wise baseline for variance-reduced REINFORCE on the
   Linear AR model. Validates Deep Think's BRAIN-rescue path's
   noise-resilience claim.

3. **Phase 3 architecture revision** (`_bmad/architecture.md`):
   distribution-learning role uses BRAIN+Linear-AR, NOT BRAIN-as-
   published. The Linear AR is a Carnot-specific extension required
   to handle k=15 AND-composition correlations.

4. **Memory entry** for the BRAIN+Linear-AR rescue construction.

### Composition with prior verdicts

Where BRAIN+Linear-AR fits in the architecture stack:

```
Inference Sampling:     THRML block-Gibbs at candidate-warm-start
                        (DT-7 + DT-MCMC-NULL + DT-MCMC-STATELESS)

Verification Cond.:     Soft-Gibbs Residual (DT-OT-RESIDUAL)

Optimization (argmin):  Carnot's existing Gibbs heuristic on unreduced HUBO
                        (DT-COMPOSITION (e) ruled out SpecAnn)

Distribution Learning:  Adaptive K-PCD with SA/PT (DT-MCMC-K1) for
                        the substrate parameter θ.
                        Optionally: BRAIN+Linear-AR REINFORCE for the
                        learned generator q_θ on noisy hardware
                        (DT-BRAIN-CORRELATIONS rescue).
                        FR-11 v15+ uses λ-GRPO patch (DT-2).

Verifier Geometry:      Mukherjee + C-parameterized extension (DT-5)
Security:               Kinetic defense-in-depth (DT-MCMC-NULL)
```

**The two distribution-learning options are orthogonal** and may both
have a role:
- Adaptive K-PCD: trains the substrate energy function θ
- BRAIN+Linear-AR: trains a separate generator q_θ for fast sampling
  on noisy hardware (e.g., KV260 POC, Extropic Z1 readiness)

### Final 5-paper sweep outcome (9 verdicts complete)

| Paper | Status | Reason |
|---|---|---|
| OT Verification | EXTENDED | C-parameterized + Soft-Gibbs Residual |
| GRPO-as-PRM | ADOPTED | λ-GRPO patch for FR-11 v15+ |
| MCMC Layers | RULED OUT | 4 verdicts: correctness, security, training stability, latency |
| BRAIN | EXTENDED | BRAIN+Linear-AR rescue (DT-BRAIN-CORRELATIONS) |
| Spectral Annealing | RULED OUT | HUBO→QUBO fatal + level-crossing brittleness |

**Three Carnot-specific extensions become paper-v6 contributions:**
1. C-parameterized OT robustified theorem (DT-5)
2. Soft-Gibbs Residual for AND-composed verifier ensembles (DT-OT-RESIDUAL)
3. BRAIN+Linear-AR REINFORCE for correlated targets (DT-BRAIN-CORRELATIONS)

Plus one Carnot-discovered security insight:
4. Kinetic defense-in-depth (DT-MCMC-NULL)

**These four novel constructions form the heart of paper-v6's
contribution beyond standard literature.**

### Striking methodology outcome

The 9-verdict sweep produced a coherent, rigorously-grounded Phase 3
architecture by interrogating each ICLR 2026 paper against Carnot's
specific structural constraints. **3 of 5 papers' algorithms were
ruled out, but every paper informed the design** — either via direct
adoption (OT, GRPO-as-PRM), constructive extension (Soft-Gibbs from
OT), or as the impetus for a Carnot-specific replacement (Linear AR
replacing BRAIN's factorized Bernoulli, adaptive K-PCD replacing
MCMC Layers' K=1, existing Gibbs replacing SpecAnn).

**The integration plan's value is in having documented both the
positive and negative results, so future planners don't re-litigate.**

---

## DT-COMPOSITION response (received 2026-05-08, ~20:30Z) — **MIXED VERDICT: VALUABLE FAILURE-MODE FINDINGS; TOP-LINE RECOMMENDATION CONTRADICTS PRIOR VERDICTS**

**Question summary:** Can MCMC Layers + BRAIN + Spectral Annealing be
composed in different sampler-pipeline roles, or do their assumptions
conflict?

**Verdict (split):**
- **Accept** Deep Think's findings on (b) BRAIN+MCMC Layers training
  incompatibility, (c) SpecAnn level-crossing failures, (e) HUBO→QUBO
  reduction fatal for SpecAnn, (f) Gadget-Induced Mean-Field Collapse
  failure mode.
- **Reject** Deep Think's top-line recommendation ("Adopt MCMC Layers
  jointly for sampling AND distribution learning"). This contradicts
  DT-7 (KL=0.17 fix requires THRML, not MH), DT-MCMC-NULL (MH amplifies
  null-space mimicry — security forbids), DT-MCMC-K1 (K=1 PCD diverges
  on non-convex Ising), and DT-MCMC-STATELESS (fresh-restart MH ≪ Gibbs
  at K=100). **Deep Think appears to have lost context across sessions.**

### Findings to ACCEPT and integrate

**(a) (I) inference and (III) distribution learning are NOT cleanly
separable** because they share state via PCD. Persistent chains in (I)
serve as initialization for negative-phase gradient updates in (III).
Decoupling severs the shared-state loop.

This is a structural insight worth keeping. **Implication for Carnot**:
the candidate-warm-start architecture (DT-MCMC-STATELESS) gives us
something interesting — at INFERENCE, the chain is initialized at the
candidate (no PCD persistence). At TRAINING, the chain persists across
gradient steps (PCD as required by DT-MCMC-K1's adaptive-K spec). These
are explicitly decoupled.

**(b) BRAIN + MCMC Layers training paradigms are STRICTLY INCOMPATIBLE.**

```
MCMC Layers Fenchel-Young K=1 guarantee: requires chain near π_{θ,t}
BRAIN REINFORCE: explicitly off-policy, mode-seeking, high-variance
                  → repeatedly knocks MCMC chains out of stationarity
                  → voids Fenchel-Young K=1 guarantees
```

Rules out a hybrid we hadn't explicitly considered. Confirms each
paper's paradigm is internally consistent but pairwise incompatible.

**(c) SpecAnn warm-start fails at phase transitions.** Davis-Kahan
gives smooth eigenspace rotation under continuous ΔJ for *most*
training steps. But at "level crossings" (first-order phase transitions
where spectral gap closes), the principal eigenvector abruptly
orthogonalizes to the new ground state. **Continuous homotopy path
shatters; SpecAnn forced into cold-restart at the worst possible
moment** (mid-training when EBM is most rapidly evolving).

This is a genuine new finding. Phase 3 substrate training will hit
phase transitions — predicted by anomalous behavior of `‖J‖_∞` per
DT-MCMC-K1 — making SpecAnn brittle exactly when we need it most.

**(d) Match-by-construction obviates noise-as-resource (in some
contexts).** This finding is correct *if* Carnot's energy is analytic
and noiseless. **Caveat for Carnot**: our energy comes from k=6
verifier ensemble outcomes, which involve LLM stochasticity, code
execution oracle calls, and probabilistic verifier outputs. Energy
IS noisy in practice. So BRAIN's variance-reduction baseline does
solve a real problem in Carnot's setting — but the OTHER objections
to BRAIN (factorized expressivity, DT-BRAIN-CORRELATIONS pending)
likely still rule it out.

**(e) HUBO→QUBO reduction is FATAL for SpecAnn on Phase 3.** Carnot's
k=15 AND-composed indicator energy is HUBO. SpecAnn requires QUBO.
Reduction needs O(k) auxiliary "gadget" variables per clause + massive
penalty weights M ≫ w_i to enforce logical consistency.

```
Penalty stiffness → eigenvalue clustering → exponentially shrinking
eigengaps → fractured continuous path → SpecAnn trapped in spurious
gadget-satisfying local minima
```

**SpecAnn is mathematically unviable for Phase 3.** Strong rule-out.

**(f) Gadget-Induced Mean-Field Collapse worst-case failure mode:**

The naive three-technique composition produces a regime *strictly
worse than status-quo Gibbs baseline*:

1. **Trap (SpecAnn)**: HUBO→QUBO injects gadgets + massive penalties
2. **Collapse (BRAIN)**: factorized Bernoulli q_θ cannot represent
   rigid gadget-base correlations → mean-field collapse → hallucinated
   invalid state updates
3. **Freeze (MCMC Layers)**: receives broken parameter updates →
   confronted by infinite-penalty walls → MH acceptance plummets to
   near-zero → chains permanently freeze

**Result**: optimizer hallucinates, sampler paralyzed, distribution
learner captures broken marginals. **Carnot's existing Gibbs sampler
on unreduced k=15 energy effortlessly avoids the entire lockup.**

This is an excellent worst-case argument for keeping the existing
Gibbs path. Codifies the Phase 3 architecture decisions.

### What I REJECT and why

**Deep Think's top-line recommendation: "Adopt MCMC Layers jointly for
sampling AND distribution learning. It resolves the inference KL=0.17
gap via its exact match-by-construction MH target."**

This DIRECTLY contradicts the prior 4 verdicts:

1. **DT-7 (vendor THRML)**: single-site MH **cannot match block-Gibbs**
   at finite K. Mixing-time parity at n=128 SK glass requires K ≫ 10^15
   sweeps. **MCMC Layers does NOT resolve KL=0.17 by construction.**
   Deep Think contradicts itself.

2. **DT-MCMC-NULL (kinetic-defense-in-depth)**: MH plateau-diffusion
   advantage is a security LIABILITY for null-space-vulnerable EBMs.
   Adopting MH at inference amplifies null-space mimicry 50%. **Forbids
   MH at inference on security grounds.**

3. **DT-MCMC-K1 (K=1 PCD diverges)**: on non-convex Ising at production
   scale (n=128, t=1), K=1 PCD freezes when `‖J‖_∞` grows during
   discriminative training. **Phase 5 needs adaptive K + SA/PT, NOT
   "joint MCMC Layers for full forward-backward pass."**

4. **DT-MCMC-STATELESS (warm-start at candidate)**: fresh-restart MH at
   K=100 is STRICTLY WORSE than fresh-restart Gibbs in low-acceptance
   regimes (wastes 95% of latency budget standing still). **Forbids MH
   at inference on latency grounds.**

**The pattern**: DT-COMPOSITION evaluated each technique on its own
mathematical merits but lost track of the empirical constraints
established by prior verdicts. Specifically, DT-COMPOSITION assumed:
- "MCMC Layers gives exact match to π" (FALSE per DT-7 — only
  asymptotically, not at production K)
- "MCMC Layers safely replaces PCD loop" (FALSE per DT-MCMC-K1 —
  K=1 PCD diverges)
- The "exact deterministic mathematical match" framing ignores the
  block-Gibbs vs single-site-MH structural divergence DT-7 proved.

**Methodology lesson**: Deep Think's verdicts can conflict across
sessions. Synthesize the architecture from ALL verdicts, not just the
latest one. The integration plan is the source of truth, not any
individual prompt response.

### The actual final architecture (synthesizing 8 verdicts honestly)

```
INFERENCE API (stateless HTTP):
  Sampler:    THRML block-Gibbs (vendored Apache-2.0)
              [DT-7: structural mismatch with single-site MH at finite K]
  Init:       user-provided candidate from { prompt, candidate } payload
              [DT-MCMC-STATELESS: avoids χ² cold-start penalty]
  K:          100 sweeps (latency budget)
  Conditioning: Soft-Gibbs Residual µ_res^β(y) ∝ µ(y) · exp(-β V(y))
              [DT-OT-RESIDUAL: Hard-BRS fails under k=15 AND-composition]
  Justified:  correctness (DT-7), security (DT-MCMC-NULL), latency
              (DT-MCMC-STATELESS), AND-composition (DT-OT-RESIDUAL)

PHASE 5 TRAINING (offline, batch):
  Sampler:    Adaptive K-PCD (Carnot-developed; not MCMC Layers)
              [DT-MCMC-K1: K=1 PCD diverges; need adaptive K + SA/PT]
  Diagnostics: Hamming Velocity + Persistent Energy Gap
  Temperature cycling: SA in-K-steps OR PT replicas
  Init:       persistent chain (PCD)
  FR-11:      v15+ uses λ-GRPO patch
              [DT-2: standard GRPO has |λ|-bug causing mode-collapse via
              anti-exploration; v14 retentions need audit]

OPTIMIZATION (argmin):
  Method:     Carnot's existing Gibbs heuristic on unreduced HUBO energy
              [DT-COMPOSITION (e): SpecAnn requires QUBO reduction; the
              required HUBO→QUBO gadgets are mathematically fatal]
  REJECTED alternative: SpecAnn (level-crossing brittleness +
              gadget-induced mean-field collapse)

VERIFICATION GEOMETRY:
  Framework:  Mukherjee OT + Carnot's C-parameterized robustified
              extension [DT-5]
  Two thresholds: C* (PI boundary), C_inv (inversion threshold)
  Coverage:   PAC-Bayes via Soft-Gibbs Z_β ≥ ∏ exp(-β α_i)

SECURITY:
  Property:   Kinetic defense-in-depth [DT-MCMC-NULL]
              Glauber Gibbs plateau-friction inflates adversary's
              compute requirement; raises C_inv
  Anchor:     k=15 AND-composition + red-team audit + tripwire
              (Phase 3 architecture per existing project record)
```

### Tasks adjusted (none new from DT-COMPOSITION; existing tasks reaffirmed)

DT-COMPOSITION's findings reinforce existing `.120 priorities. No new
tasks needed because the valuable findings are *negative* — ruling out
hybrid compositions that we hadn't actively considered:

1. ✅ **exp15ZZ-thrml-vendored-block-gibbs-replacement** (DT-7) —
   reaffirmed; DT-COMPOSITION does NOT change this.
2. ✅ **exp15PP-adversarial-pass-rate-saturation-rho-of-C** (DT-5) —
   reaffirmed.
3. ✅ **exp15QQ-phase5-pcd-divergence-audit-tiny-ising** (DT-MCMC-K1) —
   reaffirmed; DT-COMPOSITION's recommendation to "use MCMC Layers
   jointly" is REJECTED.
4. ✅ **exp15RR-phase5-adaptive-K-schedule-implementation** —
   reaffirmed (still gated on QQ).
5. ✅ **exp15TT-thrml-block-gibbs-plateau-friction-audit** (DT-MCMC-NULL).
6. ✅ **exp15UU-candidate-warm-start-vs-cold-start-benchmark**
   (DT-MCMC-STATELESS).
7. ✅ **exp15VV-soft-gibbs-residual-implementation** (DT-OT-RESIDUAL).
8. ✅ **exp15WW-soft-gibbs-coverage-bound-empirical-verification**.
9. ✅ FR-11 retention audit + v15 λ-GRPO patch (DT-2).
10. **NEW** `.121 task: explicitly add **SpecAnn rejected for Phase 3**
    to architecture decision record. Document the HUBO→QUBO + level-
    crossing reasoning in `_bmad/architecture.md` so the question
    doesn't get re-litigated.

### Implications for follow-up prompts (cascade)

- **DT-BRAIN-CORRELATIONS**: still pending. DT-COMPOSITION (b) ruled
  out BRAIN + MCMC Layers hybrid; DT-BRAIN-CORRELATIONS specifically
  asks whether BRAIN's factorized Bernoulli can model Phase 3
  correlations. If answer is no (likely per the paper's own
  limitations), BRAIN is fully ruled out — making the negative-adoption
  story complete.

### Striking outcome of the 5-paper sweep

After 8 Deep Think verdicts, **3 of 5 ICLR 2026 papers' algorithms are
RULED OUT for production adoption** in Carnot:

| Paper | Algorithm | Adoption status |
|---|---|---|
| OT Verification | Hard-BRS / SRS / SMC | EXTENDED (C-parameterized + Soft-Gibbs Residual) |
| GRPO-as-PRM | λ-GRPO patch | ADOPTED for FR-11 v15+ |
| MCMC Layers | Algorithm 1 / Algorithm 2 | RULED OUT (DT-7, DT-MCMC-NULL, DT-MCMC-K1, DT-MCMC-STATELESS) |
| BRAIN | REINFORCE on factorized Bernoulli | RULED OUT (DT-COMPOSITION; DT-BRAIN-CORRELATIONS pending) |
| Spectral Annealing | Eigenvalue homotopy | RULED OUT (DT-COMPOSITION (e), HUBO→QUBO fatal) |

The rule-outs are *informative* — each paper's analysis revealed a
constraint Carnot must respect. The integration plan's value is in
having documented the negative results, so future planners don't
re-litigate.

---

## DT-OT-RESIDUAL response (received 2026-05-08, ~20:00Z) — **VERDICT: LEMMA 3.4 SURVIVES BUT THEOREM 3.10 FAILS IN PRACTICE; CARNOT'S SOFT-GIBBS RESIDUAL IS A NOVEL EXTENSION**

**Question summary:** Does Lemma 3.4's residual identity µ_res = µ(· | Ŝ)
survive when µ is the EBT prior pushed through sgn → {-1,+1}^d and Ŝ
is k-AND-composed? Does Theorem 3.10's exponential decay still bound
SubOpt for Phase 3?

**Verdict: Lemma 3.4 survives translation cleanly. But Theorem 3.10
SURVIVES IN THEORY, FAILS IN PRACTICE — k=15 AND-composition makes
hard-BRS catastrophically unusable. Carnot's "Soft-Gibbs Residual"
(constructed by Deep Think) is a novel paper-v6 contribution that
salvages the bound.**

### Translation analysis (a)–(d)

**(a) Pushforward measurability: SURVIVES.**

`Y = {-1,+1}^d` has the trivial power-set σ-algebra. The sgn map
`sgn: [-1,1]^d → Y` is Borel-measurable (orthant preimage). The
pushforward `µ(y) = µ_z(sgn⁻¹(y))` is well-defined. Conditioning
commutes with pushforward — rejecting latent z samples outside
`sgn⁻¹(Ŝ)` is operationally identical to pushing forward to Y and
conditioning directly on Ŝ.

**(b) AND-composition zero-measure pathology: SURVIVES (the pathology
doesn't exist on discrete spaces).**

In finite discrete space, every y has strictly positive mass under any
EBT prior with full bounded support. Therefore `µ(∩_i Ŝ_i) = 0` iff
the verifiers are *logically contradictory* and the intersection is
strictly empty (`∩_i Ŝ_i = ∅`). The continuous-space pathology where
positive-measure sets intersect to form measure-zero overlap (k planes
intersecting at a line) **doesn't exist on `{-1,+1}^d`**.

If intersection is non-empty: positive mass, Lemma 3.4 holds.
If intersection is empty: `0/0` division is universally undefined —
that's a logic failure, not a measure-theoretic lemma failure.

**(c) Discrete densities: SURVIVES (and simplifies).**

Finite discrete spaces are Polish under discrete metric. PMFs act as
exact densities w.r.t. counting measure. The OT residual identity
`µ_res = µ(· | Ŝ)` relies purely on Total Variation algebraic
cancellation and holds for discrete PMFs without Radon-Nikodym
derivatives. The continuous-density framing in the original paper is
a **red herring** for Carnot's substrate.

**(d) Theorem 3.10 decay bound: SURVIVES IN THEORY, FAILS IN PRACTICE.**

```
SubOpt(BRS) = OTC(β)·(1 − 1/M)^N
where M = 1/µ(Ŝ_AND)
```

For k=15 AND-composition, `µ(Ŝ_AND)` becomes vanishingly small even
when non-empty. **M scales toward `2^O(k)`** → expected acceptance rate
plunges → operational decay bound flatlines → **BRS unusable at
test-time for Phase 3**.

The theorem is analytically true but operationally hollow. Catastrophic
infinite-loops when verifiers logically conflict.

### THE NOVEL EXTENSION (paper-v6 contribution): Soft-Gibbs Residual

Deep Think constructed a Carnot-specific residual that handles the
combinatorial collapse:

```
V(y) = Σ_{i=1..k} 1{y ∉ Ŝ_i}              # count of failed verifiers
β > 0 (Carnot hyperparameter)
µ_res^β(y) = µ(y) · exp(−β V(y)) / Z_β     # Boltzmann-softened residual
Z_β = E_µ[exp(−β V(y))]                     # partition function
```

**To sample**: draw `z ~ µ_z`, bottleneck to `y = sgn(z)`, evaluate
V(y), accept with probability `A(y) = exp(−β V(y)) ∈ (0, 1]`.

Three properties:

1. **Preserves rejection-sampling implementability.** No complex OT
   coupling solvers needed; just one prior sample + one V evaluation +
   one Bernoulli accept.

2. **Quotes a coverage bound via Jensen.** Since `f(x) = e^{-βx}` is
   convex:

   ```
   Z_β ≥ exp(−β · E_µ[V(y)]) = ∏_{i=1..k} e^{−β α_i}
   ```

   where `α_i = P_µ(y ∉ Ŝ_i)` is the individual marginal failure rate
   of verifier i. **Coverage bounded by individual verifier risks alone**
   — no joint estimation needed. PAC-Bayes-like.

3. **Handles AND-composition geometry gracefully.** When strict
   intersection is empty (`min V(y) > 0`), Z_β never crashes to zero.
   Sampler automatically concentrates probability mass on the
   minimum-violation subspace — the responses satisfying the maximum
   possible subset of verifiers. Degrades smoothly instead of failing.

### Falsifiable predicate

Setup: n=8 latent EBM prior + k=3 deliberately-contradictory verifiers
(e.g., v_1 demands y_1=+1, v_2 demands y_1=-1).

Run unmodified Hard-BRS and adapted Soft-BRS for N steps each. Track
empirical SubOpt.

Predicted falsification:
- **Hard-BRS empirical SubOpt flatlines at 1.0** (infinite rejection
  loop, exactly 0 acceptance rate) — falsifies operational translation
  of Theorem 3.10.
- **Soft-BRS empirical SubOpt decays exponentially**, tightly matching
  `≈ (1 − Z_β)^N` (the predicted theoretical trajectory).

### Tasks to file

1. **NEW `.121 task** (post-Soft-Gibbs derivation): `exp15VV-soft-gibbs-
   residual-implementation`. Implement the Soft-Gibbs residual sampler
   alongside Hard-BRS for comparison. Acceptance gate: at n=8, k=3
   contradictory verifiers, Hard-BRS flatlines + Soft-BRS decays per
   prediction.

2. **NEW `.121 task**: `exp15VV-bis-soft-gibbs-coverage-bound-empirical-
   verification`. Measure individual α_i for k=6 ensemble on Carnot's
   calibration corpus. Verify Jensen-bounded `Z_β ≥ ∏ e^{-β α_i}` holds
   empirically. Determine optimal β (tradeoff: low β → loose coverage
   but high acceptance rate; high β → tight coverage but low acceptance).

3. **Phase 3 architecture revision**: the k=15 AND-composition design
   intent is preserved BUT the sampling-time conditioning under it
   uses Soft-Gibbs Residual, not hard intersection. `_bmad/architecture.md`
   Phase 3 substrate spec must specify:
   - Inference: THRML block-Gibbs (DT-7) at candidate-warm-start
     (DT-MCMC-STATELESS), with Soft-Gibbs Residual conditioning
   - β as a per-deployment hyperparameter (paper-v6 Appendix should
     report calibration values)

4. **Paper-v6 §3 disclosure paragraph (new)**:

   > "Carnot's Phase 3 verification pipeline cannot adopt the
   > Optimal Transport framework's Hard-BRS, SRS, or SMC algorithms
   > out-of-the-box. The k=15 AND-composition reduces the joint
   > acceptance probability `µ(∩_{i=1..15} Ŝ_i)` toward 2^{-O(15)},
   > rendering Theorem 3.10's exponential decay analytically true but
   > operationally vacuous. Worse, when verifiers logically conflict,
   > Hard-BRS rejects all samples (infinite loop). We instead introduce
   > the Soft-Gibbs Residual: `µ_res^β(y) ∝ µ(y) · exp(-β V(y))` where
   > V(y) counts failed verifiers and β > 0 is a calibration parameter.
   > This residual: (i) preserves rejection-sampling at constant
   > per-step cost; (ii) admits a PAC-Bayes-like coverage bound
   > Z_β ≥ ∏_{i} e^{-β α_i} via Jensen's inequality, requiring only
   > individual verifier marginal failure rates; (iii) gracefully
   > handles empty hard intersections by concentrating on minimum-
   > violation subspace. The Soft-Gibbs Residual is, to our knowledge,
   > a novel construction in the OT-verification literature."

### Composition with prior verdicts

The verdicts now compose into a complete Phase 3 architecture:

```
PHASE 3 SUBSTRATE (Deep Think 2026-05-08, 7 verdicts):

  Inference Sampling (per request):
    THRML block-Gibbs at candidate-warm-start (DT-7 + DT-MCMC-NULL +
    DT-MCMC-STATELESS), K=100 sweeps.
    Provides: correctness (parity to reference), security (kinetic
    defense-in-depth via Glauber plateau-friction), latency.

  Verification Conditioning (per accepted sample):
    Soft-Gibbs Residual µ_res^β(y) ∝ µ(y) · exp(-β V(y)) (DT-OT-RESIDUAL).
    Replaces hard ∩_i Ŝ_i conditioning with Boltzmann-softened
    rejection. β calibrated per-deployment.

  Compute-Bounded Safety Regime:
    SubOpt(A; C) = OTC(β_OT) · (1 - α(C) · J(C)) (DT-5).
    Two thresholds: C* (PI boundary), C_inv (inversion).
    Paper-v6 reports C_max < C_inv guarantee.

  Training (offline, batch):
    Adaptive K-PCD with SA/PT (DT-MCMC-K1).
    FR-11 v15+ uses λ-GRPO patch (DT-2).
```

**Note**: there are TWO different β parameters in this stack:
- β_OT (coverage) from DT-5
- β_softgibbs (residual softness) from DT-OT-RESIDUAL

These are independent. Need to disambiguate notation in paper-v6.

### Implications for follow-up prompts (cascade)

- **DT-BRAIN-CORRELATIONS**: unchanged. Phase 3 expressivity question
  for distribution learning still pending.
- **DT-COMPOSITION**: substantially advanced. Phase 3 architecture now
  has concrete answers for inference sampling + verification conditioning;
  composition question remaining is on training-time distribution
  learning role.

---

## DT-MCMC-STATELESS response (received 2026-05-08, ~19:30Z) — **VERDICT: WARM-START AT THE CANDIDATE; DECOUPLE TRAINING SAMPLER FROM INFERENCE**

**Question summary:** Persistent-chain PCD vs. Carnot's stateless HTTP
API contract. Cold-start vs warm-start vs cached-warm-start vs
client-passed-state. Is fresh-restart MH ever strictly worse than
fresh-restart Gibbs?

**Verdict (despite the "moot" framing after DT-7/MCMC-K1/MCMC-NULL):
this prompt produced a load-bearing architectural insight I had missed.
Carnot's API payload `{ prompt, candidate }` already contains the warm
start — initialize Gibbs at the candidate.** This bypasses the χ²
cold-start penalty without any session state, no client-passed state,
no cached state.

### Key findings

**(a) Spectral-gap-parameterized TVD bound:**

```
d_TV(μ₀ Q^100, π_{θ,t}) ≤ (1/2) · √χ²(μ₀ ‖ π_{θ,t}) · (1 − γ)^100
```

| Initialization | χ² penalty | TVD at K=100 |
|---|---|---|
| Warm-start (PCD persistent) | O(1) | tightly controlled by (1−γ)^100 alone |
| Cold-start (uniform random) | O(e^{ΔE_max/2t}) — astronomical (e.g., 2^512 in 128-byte space) | hopelessly fails to mix at K=100 |

**The gap between cold and warm NEVER closes** for production K=100 —
would require γ ≈ 1 − O(e^{−D/200}) ≈ 1, which local-search MH
categorically cannot achieve.

**(b) No target-free K=1 analog.** Proposition 3's K=1 unbiasedness is
a *training-time gradient construct* requiring ground-truth `y` for
the target-dependent regularizer Ω_y. At inference, you don't compute
gradients — you sample from the primal `π_{θ,t}(· | prompt)`. There
is no mathematical mechanism to collapse mixing time to K=1 without
ground truth. Dropping Ω_y forfeits exactness and reverts to standard
asymptotic K → ∞ unbiasedness.

**(c) Cached warm-start: REJECT.** A cached state from a recent
production sample belongs to a *completely different prompt's
landscape*. Initializing the chain with out-of-distribution state
"traps the sampler in an irrelevant deep local minimum, acting as an
adversarial initialization that performs WORSE than a uniform cold
start." Counter-intuitive but correct: stale state ≠ warm state when
the conditional distribution depends on prompt.

**(d) Client-passed-state: CONDITIONALLY ACCEPT but practically
flawed.** Mathematically preserves PCD persistence (recovers Prop 5).
But only works for repeated iterative queries on the same prompt. For
zero-shot verification (Carnot's primary load — one new prompt, one
new candidate), client has no valid prior state. Forces cold-start
regime anyway.

**(e) Fresh-restart MH is STRICTLY WORSE than fresh-restart Gibbs at
K=100 in low-acceptance regimes.**

When the landscape has high correlations or low temperatures, MH's
acceptance rate α ≪ 1. With α = 0.05, MH rejects 95% of proposals →
only ~5 actual state transitions in 100 sweeps → wastes 95% of the
100ms compute budget standing still.

Gibbs analytically integrates over 1D/block conditional distributions
and samples directly → mathematically guaranteed effective coordinate
acceptance rate of 1.0 → 100 energy-lowering transitions per K=100.

**Under strict latency constraint, Gibbs vastly outperforms MH during
critical burn-in phase.** This compounds DT-MCMC-NULL's security
recommendation with a separate latency-quality argument.

### THE LOAD-BEARING INSIGHT: candidate-as-warm-start

> "Carnot's API payload is `{ prompt, candidate }`. Instead of
> complicated state-passing, initialize the Gibbs chain directly at
> the provided candidate. The user's candidate is a structurally
> valid, localized proxy for the target mode. This bypasses the
> catastrophic χ² cold-start divergence penalty, allowing Gibbs to
> greedily explore the valid local verification neighborhood without
> wasting a single sweep of your 100ms budget."

This is brilliant and obvious in retrospect. **The verifier's job is
to verify a candidate; the candidate IS the warm start.** No session
store, no client-passed state, no cached state needed. The API
contract already provides what we need.

### Final architecture (synthesizes DT-7 + DT-MCMC-K1 + DT-MCMC-NULL + DT-MCMC-STATELESS)

```
┌─────────────────────────────────────────────────────────────┐
│ Carnot's two-sampler architecture (decoupled by use case)   │
│                                                              │
│ INFERENCE API (stateless HTTP):                              │
│   Sampler:     THRML block-Gibbs (vendored Apache-2.0)       │
│   Init:        the user-provided `candidate` from payload    │
│   K:           100 sweeps (latency budget)                   │
│   Justified:   correctness (DT-7), security (DT-MCMC-NULL),  │
│                latency (DT-MCMC-STATELESS), warm-start       │
│                                                              │
│ PHASE 5 TRAINING (offline, batch):                           │
│   Sampler:     adaptive K-PCD with SA/PT (DT-MCMC-K1)        │
│                may use MH layer for FY gradient (Prop 3)     │
│   K:           dynamically scheduled per chain pathology     │
│   Diagnostics: Hamming Velocity + Persistent Energy Gap      │
│   Init:        persistent chain (PCD per Prop 5)             │
│                                                              │
│ Decoupling is mathematically valid: both target same π_{θ,t} │
│ (DT-MCMC-STATELESS explicit confirmation).                   │
└─────────────────────────────────────────────────────────────┘
```

This is the **complete sampler-side architecture for Carnot** after
five Deep Think verdicts. No further questions about samplers in
isolation are open — only cross-paper composition (DT-COMPOSITION) and
substrate-measurability (DT-OT-RESIDUAL).

### Tasks to file

1. **UPDATE** `.120 task seed `exp15ZZ-thrml-vendored-block-gibbs-
   replacement`: add specification that the inference Gibbs chain MUST
   initialize at the user-provided candidate (not at random state, not
   from cache). This is a Carnot-specific design constraint on top of
   the THRML vendoring.

2. **NEW** `.121 task seed `exp15UU-candidate-warm-start-vs-cold-start-
   benchmark`: empirically measure inference latency and verification
   quality for (a) candidate-warm-start (b) cold-start (c) cached-state
   warm-start at K ∈ {10, 50, 100, 500} sweeps. Confirms the χ² penalty
   prediction empirically and rules out cached-state as a deployment
   pattern.

3. **Phase 5 architecture revision** (`_bmad/architecture.md`): the
   in-situ training spec must explicitly note that **the training
   sampler can differ from the inference sampler**. Both target π_{θ,t}.
   Training: differentiable MH layer (PCD, adaptive K, SA/PT).
   Inference: fresh-restart Gibbs initialized at candidate.

4. **Paper-v6 §3 disclosure (revised)**: add the candidate-as-warm-
   start design choice as a Carnot-specific contribution. Draft text:

   > "Carnot decouples its training-time and inference-time samplers.
   > Training uses a differentiable MCMC layer with adaptive K-PCD and
   > simulated-annealing temperature cycling [Sullivan 2026, K-schedule
   > rationale per Appendix Y]. Inference uses fresh-restart block-Gibbs
   > [vendored from Extropic THRML 0.1.3], initialized at the user-
   > provided candidate. This exploits the verifier API contract:
   > the candidate `y_init = candidate` is a structurally valid,
   > prompt-conditioned proxy for π_{θ,t}, bypassing the χ² cold-start
   > penalty `√χ²(μ₀ ‖ π_{θ,t}) ≈ O(e^{ΔE_max/2t})` that would
   > otherwise dominate the K=100 latency budget. Cached-state and
   > client-passed-state alternatives were rejected: cached state from
   > a different prompt is mathematically equivalent to adversarial
   > initialization (Deep Think 2026-05-08)."

### Methodological note (worth recording)

I had marked DT-MCMC-STATELESS "completely moot" after the
DT-7/MCMC-K1/MCMC-NULL cascade eliminated MCMC Layers from inference.
Sending the prompt anyway surfaced the candidate-warm-start
architectural insight that would otherwise have been missed.

**Lesson**: even when a prompt's headline question becomes moot, the
prompt's *secondary* questions (in this case (e) on fresh-restart MH
vs Gibbs and the API-contract reasoning) can produce load-bearing
insights. Send all drafted prompts even if their headline is
already-resolved.

### Cascade for follow-up prompts

- **DT-OT-RESIDUAL**: unchanged. Substrate-measurability is orthogonal
  to sampler architecture.
- **DT-BRAIN-CORRELATIONS**: unchanged. Phase-3 expressivity question
  still gates BRAIN's role in distribution learning.
- **DT-COMPOSITION**: simplified by candidate-warm-start design. The
  three-sampler composition becomes:
  - (I) Inference: fresh-restart THRML block-Gibbs at candidate
        (DT-7 + DT-MCMC-NULL + DT-MCMC-STATELESS)
  - (II) Argmin: Spectral Annealing (still pending DT-COMPOSITION)
  - (III) Distribution learning: BRAIN's REINFORCE OR adaptive-K PCD
        with SA/PT (DT-MCMC-K1 + DT-BRAIN-CORRELATIONS pending)

---

## DT-MCMC-NULL response (received 2026-05-08, ~19:00Z) — **VERDICT: STICK WITH GIBBS — ITS SLUGGISHNESS IS A SECURITY FEATURE**

**Question summary:** Does MH proposal correction concentrate on the
joint null space N (Spera 9.2) faster than current Gibbs? Could
adopting MCMC Layers in Phase 3 amplify the null-space-mimicry attack?

**Verdict: YES, MH amplifies the attack. The recommendation is to
stick with Gibbs and DOCUMENT THE SLUGGISHNESS AS A SECURITY FEATURE.**
This is a paper-v6-publishable insight: **kinetic-defense-in-depth**.

### Key findings

**(a) MH concentrates on N strictly faster than Gibbs at finite K.**
Peskun's Theorem + acceptance mechanics:

| Move type | MH | Gibbs (heat-bath) |
|---|---|---|
| Downhill (Δ > 0) | accepts with p = 1 | accepts with sigmoid(Δ/t) < 1 |
| **Flat plateau (Δ = 0)** | **accepts with p = 1.0** | **rejects 50% of the time** |

MH explores flat combinatorial plateaus **2× faster** than Gibbs.
Both target the same π asymptotically, but in finite-K windows,
`P_MH^(K)(N) ≫ P_Gibbs^(K)(N)`.

**(b) Carnot's landscape is plateau-dominated.** AND-composed verifiers
create a topography of:
- Wide flat plateaus where verifier-unmet states have zero gradient
- Sparse, spectrally isolated targets (verifier-block satisfaction)

The bottleneck for adversaries is plateau traversal. MH's superior
plateau-diffusion speed means it finds isolated null-space vulnerabilities
**strictly faster**.

**(c) Mixed-neighborhood (Algorithm 2) makes it STRICTLY WORSE.**
Localized null spaces are frequently protected by Hamming-1 energy
barriers (e.g., satisfying verifier 15 might temporarily require breaking
another constraint). Single-site chains naturally stall in these benign
local minima. **Algorithm 2's Hamming-k jumps allow the chain to bypass
these localized penalties entirely, tunneling directly across plateaus
and barriers into N.** Direct optimization shortcut for the adversary.

**(d) Red-Team audit becomes UNSOUND under MH.** Defender faces a
universal coverage problem (patch entire combinatorial manifold of N,
which is coNP-complete to enumerate per Spera 9.2). Attacker faces an
existential problem (need ONE unpatched y ∈ N). **If production upgrades
to MH, the reduction in search time disproportionately helps the
attacker, who can weaponize Carnot's own optimized inference engine
to find unpatched holes using negligible test-time compute.**

**(e) Null-space-aware MH proposal modification mathematically conflicts
with the Fenchel-Young loss guarantee.** Two cases:

- **With Hastings correction**: ratio q(y,y')/q(y',y) perfectly
  counteracts your bias; stationary distribution still concentrates on
  N. You merely cripple mixing speed without defending against the attack.
- **Without Hastings correction**: break detailed balance, target an
  invalid distribution, mathematically void the FY gradient estimator.

Detailed balance + null-space-defense are FUNDAMENTALLY IRRECONCILABLE
under standard MH machinery.

### Falsifiable predicate (synthetic isolation test)

Construct n=64 binary Ising representing k=15 AND-composed verifiers:
- 15 independent structural blocks of 4 bits
- E(y) = −10 · Σ_{i=1..15} **1**{block_i = target}
- Remaining 4 bits are free, planting null space N of size 2^4 at
  global minimum

Initialize 10,000 chains at y = {0}^64 (massive plateau, all verifiers
unmet). Run single-site Gibbs + Algorithm 1 MH at t=1.0.

**Mean hitting time for a single 4-bit block to traverse its plateau:**
- MH: ≈ 21.3 steps
- Gibbs: ≈ 32.9 steps (50% slower)

Composed over 15 blocks, P_MH^(K)(N) shows massive pre-asymptotic
surge, crossing the 10% mass threshold at substantially lower K than
Gibbs.

### Recommendation: Stick with Gibbs (Conservative — but for a publishable reason)

> "Carnot must rely on the inherent sluggishness of Glauber Gibbs. Its
> algorithmic inefficiency on plateaus acts as 'computational friction'
> — a kinetic defense-in-depth moat that dramatically inflates the
> test-time inference compute an adversary requires to successfully
> exploit residual null spaces."

**This is a paper-v6 publishable insight.** The "kinetic-defense-in-depth"
framing is novel: a sampler's algorithmic inefficiency on flat plateaus
becomes a security property when combined with combinatorially-detected
joint null spaces (Spera 9.2). Carnot can claim this as a contribution
on top of standard MCMC literature.

### Composition with prior verdicts

DT-7 (vendor THRML) and DT-MCMC-NULL compose ELEGANTLY:

- **DT-7**: vendor THRML's block-Gibbs to fix KL=0.17 mismatch
  (correctness)
- **DT-MCMC-NULL**: Gibbs-class samplers' plateau-friction is a
  security feature against null-space mimicry (security)

**The single recommendation is: vendor THRML block-Gibbs. Justified
by both correctness AND security.** No tension between the two paths.

### CRITICAL FOLLOW-UP: does THRML block-Gibbs inherit Gibbs's
plateau-friction property?

Block-Gibbs is multi-site (parallel update of independent vertex sets
via graph coloring) but per-bit sampling is heat-bath conditional —
NOT proposal-and-accept like Algorithm 2. At a flat plateau, each bit
in a colored block samples from sigmoid(0) = 0.5 independently → the
per-bit randomization rate is identical to single-site Gibbs.

**The difference between block-Gibbs and single-site Gibbs is
parallelism (compute speed), NOT mixing speed at the per-bit level.**
Therefore block-Gibbs SHOULD inherit single-site Gibbs's
plateau-friction security property.

**This claim needs explicit Deep Think confirmation.** Recommended
follow-up prompt: DT-MCMC-NULL-BLOCK-GIBBS — does THRML's block-Gibbs
implementation inherit the kinetic-defense-in-depth property from
single-site Gibbs, or does the parallel-block update structure create
a new attack surface?

### Tasks to file

1. **NEW `.121 task** (post-vendoring): `exp15TT-thrml-block-gibbs-
   plateau-friction-audit`. Run the synthetic null-space isolation
   test (n=64, k=15 4-bit blocks) on THRML's block-Gibbs vs Carnot's
   current single-site Gibbs vs (hypothetically) MCMC Layers Algorithm
   1. Measure mean hitting time to N at K=1..1000 sweeps. Acceptance
   gate: THRML block-Gibbs hitting time ≥ single-site Gibbs's hitting
   time (security parity). If THRML is faster than single-site Gibbs,
   audit whether the parallel-block update creates a new attack
   surface and document.

2. **Paper-v6 §3 sampler section** (`.121 task): add the kinetic-
   defense-in-depth callout. Draft text:

   > "Carnot's substrate uses block-Gibbs sampling (vendored from
   > Extropic THRML 0.1.3) rather than Metropolis-Hastings or learned
   > differentiable MCMC layers. While MH provides faster mixing to
   > stationarity, on AND-composed verifier energies — which are
   > dominated by flat plateaus with sparse spectrally-isolated null
   > regions — MH's plateau-traversal advantage becomes a security
   > liability: it accelerates adversarial concentration on the
   > coNP-complete-detectable joint null space (Spera Theorem 9.2).
   > Glauber-class samplers' algorithmic inefficiency on plateaus
   > acts as kinetic defense-in-depth: an attacker exploiting residual
   > null spaces requires Ω(plateau-diffusion-time) inference compute,
   > which Glauber's heat-bath rejection rate doubles relative to MH."

3. **Phase 3 architecture entry** (per `_bmad/architecture.md`): add
   "Sampler choice: block-Gibbs is the security-required minimum.
   MH or learned MCMC layers are NOT acceptable for production
   substrate without a constructive null-space-empty proof."

4. **Add to memory**: kinetic-defense-in-depth as a Carnot-discovered
   security property of sampler choice for null-space-vulnerable EBMs.

### Cascade for follow-up prompts

- **DT-MCMC-STATELESS**: now COMPLETELY MOOT. MCMC Layers eliminated
  from inference (DT-7), production-scale Phase 5 (DT-MCMC-K1), and
  Phase 3 substrate (DT-MCMC-NULL). The only remaining role for MCMC
  Layers in Carnot is theoretical-foundation for Phase 5 K-schedule
  reasoning.
- **DT-OT-RESIDUAL**: unchanged (substrate-measurability question is
  orthogonal to sampler choice).
- **DT-BRAIN-CORRELATIONS**: unchanged.
- **DT-COMPOSITION**: revised. Three-sampler composition simplified
  to:
  - (I) Inference sampling: THRML block-Gibbs (DT-7 + DT-MCMC-NULL)
  - (II) Optimization (argmin): Spectral Annealing (DT-COMPOSITION
    pending)
  - (III) Distribution learning at training: BRAIN's REINFORCE (gated
    on DT-BRAIN-CORRELATIONS) OR Carnot-specific PCD with adaptive
    K + SA/PT (DT-MCMC-K1 explicit recommendation)

---

## DT-MCMC-K1 response (received 2026-05-08, ~18:30Z) — **VERDICT: K=1 PCD DIVERGES ON NON-CONVEX ISING; NEED ADAPTIVE K + SA/PT**

**Question summary:** Is MCMC Layers' K=1 Fenchel-Young unbiased gradient
practically sufficient for Carnot's Phase 5 in-situ training at n=128
production scale, or does it need a K-schedule?

**Verdict: Carnot's premise contains a load-bearing CATEGORY ERROR.**
We've been conflating CD-1 guarantees (where MCMC Layers' K=1 holds)
with PCD mechanics (Carnot's actual training regime). On a non-convex
moving target, K=1 PCD **diverges** — the chains freeze and carve
"ghost modes" exactly when the verifier starts learning. Phase 5
needs adaptive K + temperature-cycling.

### Key findings

**(a) Quantitative bias model — AR(1) tracking lag:**

```
Bias(s) ≈ O((η / γ(n, t, ‖J_s‖)) · ‖∂E_π[Y]/∂θ‖)
       ≈ O(η · exp(c · √n · ‖J_s‖_∞ / t))
```

For n=128, t=1: barrier scale `√128 ≈ 11.3`. Once `‖J_s‖_∞` reaches
O(1) during training, γ collapses to **~10⁻⁵**. Since η ≫ γ, parameter
updates vastly outpace mixing time. **Persistent chains freeze,
decouple from target distribution, carve massive spurious local
minima ("ghost modes").**

**Verdict: BIAS-DOMINATED REGIME. Training diverges.**

**(b) Per-step ≠ converged-distribution unbiasedness:**

Proposition 3's K=1 unbiased gradient is for a **local surrogate loss
anchored at starting state y**, evaluated exclusively over the
Hamming-1 marginal polytope. The paper sidesteps the Sutskever-Tieleman
impossibility by **changing the objective**, not solving the original.

**The PCD mismatch (load-bearing):** To inherit the FY guarantee, you
must evaluate the loss centered at `y_data` (CD-1). Carnot's PCD starts
each K=1 step at `y_persistent`. **You're computing a perfectly
unbiased gradient for an FY loss anchored to a wandering hallucination.**

**(e) Non-convexity addressed:** the FY loss only looks 1-hop; its
mathematical guarantee survives global non-convexity *by ignoring it*.
Provides an unbiased gradient pointing to the bottom of whatever local
trap the persistent chain is in. Useless for global EBM convergence.

**(c) Discriminative training makes mixing EXPONENTIALLY WORSE:**

Carnot trains J to confidently separate valid from invalid candidates →
carves deep energy wells for valid candidates, erects massive barriers
around invalid ones → sharpens the spectrum of J → rapidly inflates
`‖J_s‖_∞`. **A K=1 chain that appears healthy at step 0 hits a
catastrophic mixing wall exactly when the verifier begins succeeding.**

This is a counter-intuitive result worth highlighting: **training
quality and chain mixing are inversely correlated** under standard PCD.

**(d) Practical K-schedule:**

| Phase | Strategy |
|---|---|
| Early (warm-up) | K=1 while J near 0; spectral gap wide |
| Mid | Scale K dynamically 5 → 20 to compute ceiling |
| Late | K capped → **must inject noise**: Simulated Annealing within K steps (spike to t=5 then cool to t=1), OR Parallel Tempering across replicas |

**Linearly raising K cannot overcome an exponential gap.** Late-phase
training requires explicit barrier-bridging via SA or PT.

**Checkable signals (do NOT monitor gradient variance — it stays
artificially stable when chains freeze):**

1. **Hamming Velocity**: `Δ_H = Hamming(y_p^(s), y_p^(s−50))`. If
   chain flips fewer than 2-3 bits over 50 gradient steps, **chain is
   hard-frozen** in local minimum. Raise K.
2. **Persistent Energy Gap**: `ΔE = E[E(y_persistent)] − E[E(y_data)]`.
   In healthy PCD, ΔE ≈ 0. If ΔE rises and plateaus high → locked out
   by barriers. If ΔE drops deeply negative → found unpenalized
   spurious mode (ghost mode).

### Falsifiable empirical predicate (cosine-similarity audit)

Use Carnot's existing exp1503/1504 tiny-Ising parity infrastructure:

1. Train identical Carnot parity models at n ∈ {4, 8, 16, 24, 32} using
   K=1 Phase 5 update rule.
2. **Oracle**: at every epoch, pause training and analytically compute
   the **exact global MLE gradient** ∇_MLE via brute-force state
   enumeration of the 2^n partition function (trivial up to n=25).
3. **Metric**: plot cosine similarity between K=1 PCD gradient and
   true MLE gradient over time.
4. **Predicted falsification**:
   - For n=4 and n=8: cosine similarity ≈ 1.0 (robust)
   - For n ≥ 16: cosine similarity **permanently crashes toward zero
     precisely at the epoch where verifier loss drops and discriminative
     `‖J‖_∞` grows**
   - Exact enumeration will reveal **K=1 PCD is actively spawning
     massive "ghost modes"** (hallucinated minima) because local K=1
     updates are blind to global probability mass

This is the cheap, definitive test. If cosine similarity crashes at
n=16, we have proof of failure mode before scaling to n=128.

### Tasks to file

1. **NEW `.120 task** (highest priority): `exp15QQ-phase5-pcd-divergence-
   audit-tiny-ising`. Run the cosine-similarity audit on n=4..32. If
   prediction confirmed, Phase 5 architecture rewrites are NEEDED.

2. **NEW `.121 task** (gated on .120 audit): `exp15RR-phase5-adaptive-K-
   schedule-implementation`. Implement adaptive K with Hamming-Velocity
   + Persistent-Energy-Gap monitoring; SA temperature-cycling and/or
   Parallel Tempering for late-phase training.

3. **Phase 5 architecture revision**: the in-situ training spec at
   `_bmad/architecture.md` Phase 5 section assumes K=1 PCD on the
   final substrate. Per DT-MCMC-K1, this is structurally insufficient.
   Phase 5 architecture must specify:
   - Adaptive K schedule
   - Temperature-cycling protocol (SA in-K-steps vs PT replicas)
   - Hamming-Velocity + Energy-Gap diagnostics as gating signals
   - Acceptance gate: training proceeds only if cosine-similarity-to-
     enumeration > 0.8 at every checkpoint where enumeration is feasible
     (likely n ≤ 25 in production)

4. **Paper-v6 §3 disclosure (revised)**: "Carnot's Phase 5 in-situ
   training uses adaptive K-PCD with temperature-cycling. K=1
   simplifications, while theoretically attractive (Sullivan 2026's
   Fenchel-Young guarantee), structurally diverge on non-convex Ising
   with discriminative training, per the cosine-similarity audit in
   Appendix Y."

### Implications for Carnot's research record

- **Counter-intuitive insight (worth a paper-v6 callout)**: training
  quality and chain mixing are inversely correlated under standard PCD.
  Carnot's Phase 5 architecture must explicitly engineer for this.
- **Prior `_bmad/architecture.md` Phase 5 spec assumed K=1 sufficiency.**
  Revise.
- **MCMC Layers' role in Carnot pipeline**: now bounded to
  - Inference sampling: ELIMINATED (DT-7, vendor THRML)
  - Phase 5 training (CD-1, K=1): VIABLE but not for production
  - Phase 5 training (PCD, K=1): **NOT VIABLE** at production scale
  - Useful only as building block: the FY loss is fine on local
    surrogates; problem is using it as global MLE proxy

### Cascade for follow-up prompts

- **DT-MCMC-NULL** (security): unchanged — still asks about MH proposal
  correction concentrating on null space. May be moot if MCMC Layers
  is also disqualified for Phase 5, but the question is structurally
  independent.
- **DT-MCMC-STATELESS**: even more moot — MCMC Layers is now eliminated
  from both inference (DT-7) AND production-scale Phase 5 (DT-MCMC-K1).
- **DT-COMPOSITION**: revised. Distribution-learning role for MCMC
  Layers needs replacement. Candidates: BRAIN's REINFORCE (if Phase 3
  expressivity question DT-BRAIN-CORRELATIONS resolves favorably),
  Spectral Annealing for argmin only, or a Carnot-specific PCD with
  adaptive K + SA/PT (the path DT-MCMC-K1 explicitly recommends).

---

## DT-2 response (received 2026-05-08, ~18:00Z) — **VERDICT: HYPOTHESIS INVERTED — RETENTIONS ARE THE BUG, NOT RETIREMENTS**

**Question summary:** Does Sullivan's `|λ|`-bug cause spurious FR-11 v14
retirements? Predicate: ≥20% flip rate when re-scoring under λ-GRPO
falsifies v14's retirement decisions.

**Verdict: NO — operator hypothesis is mathematically impossible under
AND-composition.** But the bug exists in the opposite direction:
v14's *retained* policies are likely polluted with mode-collapsed
overfit candidates. Apply λ-GRPO to v15 immediately, but for a
different reason than we expected.

### Key results

**(a) Quantitative flip-rate model:**

```
F ≈ C · P(S_G ≥ 2) · P(R(λ) < r_mean(G) | g* ∈ λ) · E[(|λ| − 1) · |Â(λ)|]
```

For F ≥ 20%, the corpus must produce groups with multiple successful
trajectories distributed across different prefixes, intermediate median
`|λ|`, and high within-group reward variance.

**(b) Flip rate is NON-MONOTONE in median |λ|** (inverted-U):

| |λ| regime | Behavior |
|---|---|
| `|λ| → 1` (diffuse) | Multiplier = 1. Standard GRPO ≡ λ-GRPO. Bug impact = 0 |
| `|λ| ≈ k/2` (intermediate) | **Maximally destructive.** Prefix isolates a subset, severe negative advantage, large multiplier |
| `|λ| → k` (saturated) | `R(λ) = r_mean(G)` ⇒ `Â(λ) = 0`. Multiplier scales zero gradient. **Bug neutralizes itself** |

**(c) Transfer to FR-11 (overestimates flip rate):**

- **Code-repair corpus** forces immense shared boilerplate (imports,
  signatures, prompt echoes) → median `|λ|` pushed toward k → bug
  naturally mitigates
- **k=8 vs Sullivan's k=36** caps max catastrophic multiplier at 7×
  vs 35×
- **7B-32B models** have lower token entropy → tighter prefix
  concentration → further mitigation

**(d) MATHEMATICAL IMPOSSIBILITY OF OPERATOR HYPOTHESIS** under
AND-composition:

Operator theorized: "a single completion finds a clever fix but its
prefix gets down-weighted because the rest failed."

**Algebraic proof of impossibility:**

```
Let r_c > 0 = reward of single clever fix.
AND-zero condition: other k−1 completions get r = 0.
Shared prefix λ containing the fix:  R(λ) = r_c / |λ|
Group mean:                            r_mean(G) = r_c / k
Constraint: |λ| ≤ k (prefix is subset of group)
Therefore:  R(λ) ≥ r_mean(G)
           ⇒ Â(λ) ≥ 0
```

**A solitary clever fix in a sea of failures is ALWAYS up-weighted.**
Mathematically barred from receiving negative advantage.

For anti-exploitation to fire, S_G ≥ 2 (multiple distinct successes
pulling r_mean above R(λ)). Under AND-composition's sparsity,
S_G ≥ 2 is exceptionally rare for borderline models.

**(e) THE SECOND-ORDER FINDING — spurious RETENTION via anti-exploration:**

The bug structurally distorts the training landscape **in both
directions**. The neglected direction is the load-bearing one:

```
Mediocre safe prefix slightly beats group mean ⇒ Â(λ) > 0
Standard GRPO multiplies positive gradient by |λ|
Rapid MODE-COLLAPSE onto safe boilerplate
Artificially inflated training rewards
```

**Carnot's RETAINED v14 manifest is likely polluted with overfit,
low-entropy models that survived the gate by exploiting the bug.**

This is a more concerning failure mode than what we hypothesized:
- Hypothesis: we were *retiring good policies* (false alarm)
- Reality: we are *retaining mode-collapsed degenerate policies*
  (silent quality regression)

### Falsifiable experiment + sample size

**For the (now-disproven) hypothesis** (re-run retired candidates):

| Test | N |
|---|---|
| 80% power, Cohen's h=0.2, two-tailed Z | 196 |
| Same, one-tailed | 155 |
| Detecting 20% flip rate vs 5% baseline noise floor (exact binomial) | 27 |

**For the actual problem** (audit retained v14 policies for
mode-collapse): see "Tasks to file" below — this is a different
experiment shape.

### Recommendation: Option (iii) + immediate λ-GRPO for v15

> Carnot should choose (iii): Accept v14 retirements as-is and document
> the `|λ|`-bug caveat in paper-v6. The operators' subjective fear of
> "lone-genius suppression" is mathematically invalid under FR-11's
> AND-composition sparsity. Retraining retired candidates is misallocation
> of compute.
>
> However, you should immediately apply the zero-overhead λ-GRPO patch
> for v15 to cure the unseen Spurious Retentions (mode-collapse via
> anti-exploration) that the bug is currently injecting into accepted
> policies.

### Tasks to file

1. **DROP** the originally-filed `.120 task seed
   `exp15XX-iclr26-grpo-secretly-prm-audit` (was: re-run retired v14
   under λ-GRPO). Mathematical impossibility proof renders this
   redundant.

2. **NEW `.120 task**: `exp15XX-fr11-v14-retained-mode-collapse-audit`.
   Audit v14 RETAINED policies for the spurious-retention failure
   mode. Acceptance metrics:
   - Token-entropy distribution vs pre-RL checkpoint (predicts
     drop ≥ 0.5 nats per token if mode-collapse)
   - Boilerplate-fraction in generated repairs (predicts ≥30% of
     output is template-recycle)
   - Per-group reward variance (predicts collapse to single-mode
     reward distribution)
   - On a held-out adversarial test set, retained-v14 vs pre-RL
     baseline accuracy (predicts retained models WORSE than pre-RL
     on out-of-distribution adversarial code)

3. **NEW `.120 task**: `exp15YY-fr11-v15-lambda-grpo-patch`. Implement
   the one-line `|λ|`-correction in TRL's GRPO trainer; train one
   FR-11 v15 candidate from current pre-RL checkpoint. Acceptance
   gate: token-entropy preserved relative to baseline at ≥90%; no
   accuracy regression vs v14 (or honest verdict identifying why);
   reduced boilerplate-fraction.

4. **Paper-v6 §3 disclosure paragraph**: "FR-11 v12-v14 trained with
   standard GRPO. v15+ adopts λ-GRPO (Sullivan 2026) to prevent
   mode-collapse via anti-exploration. v14-retained policies were
   audited for boilerplate overfit; results in Appendix X."

### Implications for Carnot's research record

- **`project_continuous_improvement.md` memory entry update**: the
  retention-quality dimension was previously implicit in v14's
  positive-utility-or-retire gate. Now explicit: v14's gate measures
  utility on a *biased* training distribution. The fix is upstream
  in training, not in the gate logic.
- **Phase 4 active-inference track relevance**: anti-exploration =
  precision-weighting suppressing high-entropy explorations of model
  prior. There may be a clean Phase-4 reformulation of this fix in
  free-energy-principle terms.
- **paper-v6 honest-results discipline**: the "lone genius suppression"
  intuition was wrong — but it surfaced a real bug (mode-collapse
  retention). Document the inversion in the limitations section as
  an example of how operator intuition needed mathematical scrutiny.

---

## DT-5 response (received 2026-05-08, ~17:30Z) — **VERDICT: PAPER-V6 PUBLISHES THE C-PARAMETERIZED VERSION**

**Question summary:** Does OT Theorem 3.6 (`SubOpt = OTC(β)·(1−αJ)`)
survive when Q11 TSS makes Youden's J compute-dependent: J → J(C)?

**Verdict: Theorem 3.6 holds pointwise. C-parameterized version is the
paper-v6 contribution.** Far from being unpublishable, this is a
"massive theoretical upgrade" that bridges abstract geometric OT with
deterministic computational scaling laws.

### Key results

**(a) Closed-form preservation:** Because Theorem 3.6's geometric OT
derivation holds pointwise for any valid (TPR, FPR) operating point,
the closed form is preserved — explicitly parameterized by attacker
compute C:

```
SubOpt(A; C) = OTC(β) · (1 − α(C) · J(C))
J(C) = TPR − (FPR_iid + ρ(C))
s_ver(C) = s_r⋆·TPR + (1−s_r⋆)·(FPR_iid + ρ(C))
α(C) evaluates dynamically based on β, s_r⋆, s_ver(C)
```

**(b) TWO critical compute thresholds — not one:**

```
C*  = ρ⁻¹((s_r⋆ · FNR / (1−s_r⋆)) − FPR_iid)         ← PI regime boundary
C_inv = ρ⁻¹(TPR − FPR_iid)                            ← inversion threshold
```

with C* < C_inv strict.

- For C < C*: verifier in healthy PI regime
- For C* < C < C_inv: PI regime LOSING efficiency but α decays while J
  stays positive; verifier still marginally helpful (`SubOpt < OTC`)
- **For C > C_inv: J(C) drops below 0; verifier is ACTIVELY HARMFUL**

**(c) Inversion mechanism (the sharp warning result):** at C > C_inv,
J(C) < 0 while α(C) stays positive ⇒ α·J becomes negative ⇒ `(1−αJ) > 1`
⇒ `SubOpt(A; C) > OTC(β)`. **Test-time verification mathematically acts
as an adversarial sieve, concentrating structurally-optimized incorrect
answers and driving final accepted distribution's accuracy STRICTLY
LOWER than the unfiltered base generator.**

This is the Carnot-specific publishable warning: "verify-and-resample"
loops have a compute-bounded safety regime, and crossing C_inv
inverts the loop into an attack vector.

**(d) OTC(β) is C-immune:** pure geometric property of base generator µ
via `s_r⋆` and coverage parameter β. Q11 TSS spoofs the verifier's
acceptance boundaries (inflates FPR) but cannot alter the theoretical
maximum correctness within the β-ball.

**(e) Worst-case-over-C bound is monotone and trivial:**

```
sup_{C ∈ [0, C_max]} SubOpt(A; C) = OTC(β) · (1 − α(C_max) · J(C_max))
```

Because ρ(C) is monotonically increasing in C (the sgn(z) bottleneck
gets weaker with more compute), the sup is at C_max. No interior
derivatives needed.

### Falsifiable empirical predicate (Adversarial Pass-Rate Saturation)

1. Isolate a large holdout corpus of base generator queries known to be
   *completely* oracle-incorrect (y ∉ S⋆).
2. Run continuous Q11 TSS structural optimization against these
   targets, sweeping discrete compute budgets C ∈ {C_1, C_2, ..., C_k}.
3. Pass optimized responses through k=6 AND-ensemble. Measure empirical
   pass-rate curve FPR_AND(C). Slope = `dρ/dC`.
4. **Falsification trigger**: at empirical C > C_inv, run full SRS
   pipeline. If end-to-end accepted accuracy drops below s_r⋆, this
   empirically validates the negative inversion bound and proves
   verify-and-resample has devolved into an attack vector.

### Recommendation: paper-v6 MUST NOT adopt Theorem 3.6 verbatim

Treating Youden's J as a static iid calibration constant is
"analytically vacuous under Carnot's threat model" — yields dangerously
optimistic safety guarantees that break in real-world deployment.

**Paper-v6 quotes the C-parameterized robustified version.** This:
- Formally certifies how much adversarial compute the verification
  ensemble can withstand (C_max < C_inv) before test-time safety
  pipelines theoretically collapse.
- Bridges abstract geometric OT limit with deterministic computational
  scaling laws — a contribution beyond Mukherjee.
- Anchors paper-v6's safety story in a falsifiable, empirically-checkable
  framework.

### Tasks to file

1. **`.120 priority update**: add a new task seed —
   `exp15PP-adversarial-pass-rate-saturation-ρ(C)-measurement`. Maps
   `dρ/dC` empirically for the k=6 ensemble, identifies empirical C*
   and C_inv. Acceptance gate: ρ(C) curve is fitted with R²>0.9; C*
   and C_inv values reported with confidence intervals.

2. **Paper-v6 §3 rewrite scope** (`.121 task): add the C-parameterized
   robustified theorem as Carnot's contribution on top of Mukherjee's
   Theorem 3.6. Notation: explicit J(C), s_ver(C), α(C); the C* and
   C_inv definitions; the "verifier as adversarial sieve at C > C_inv"
   warning result. Cite Mukherjee for the base framework, Q11 TSS
   record (`project_q11_tss_and_ste_attack.md`) for the threat model.

3. **Paper-v6 §6 (Limitations) entry** (`.121 task): document that all
   safety claims hold for C < C_max where C_max is empirically reported,
   and that operations beyond C_max have not been validated.

4. **Phase 3 architecture impact**: the k=15 AND-composition design
   intent (per `project_phase3_architecture_complete.md`) was to push
   C_inv higher. The C-parameterized framework gives a quantitative
   target: design k such that C_inv > realistic attacker compute budgets.

### Implications for follow-up prompts (cascade)

- **DT-2** (FR-11 v14 retirement under λ-GRPO): unchanged — independent
  question about training-time policy gradient bug.
- **DT-MCMC-NULL** (null-space mimicry): RELATED but independent.
  Null-space mimicry is one *mechanism* by which an attacker can
  inflate FPR; Q11 TSS via sgn(z) bottleneck is another. Both feed
  into ρ(C). Worth asking Deep Think to relate the two.
- **DT-OT-RESIDUAL** (Lemma 3.4 for Phase 3): unchanged — orthogonal
  question about substrate measurability.
- **DT-COMPOSITION**: paper-v6's safety-regime framework applies to
  whichever sampler is in (I); the C-parameterized bound is sampler-
  agnostic.

### Paper-v6 contribution status (after DT-7 + DT-5)

After two Deep Think verdicts:
- §3 verifier framework: adopt Mukherjee's notation; publish
  C-parameterized extension as Carnot's contribution
- §3 sampler: vendor THRML's block-Gibbs (DT-7); KL=0.17 finding
  resolved by construction
- §3 limitations: C-bound deployment caveat (DT-5)
- §3 mathematics: closed-form `C* = ρ⁻¹(s_r⋆·FNR/(1−s_r⋆) − FPR_iid)`
  and `C_inv = ρ⁻¹(TPR − FPR_iid)`

This is real progress on paper-v6's headline claim shape.

---

## DT-7 response (received 2026-05-08, ~17:00Z) — **VERDICT: VENDOR THRML DIRECTLY**

**Question summary:** Can MCMC Layers' single-site MH (Algorithm 1) match
THRML's block-Gibbs sampling at finite K, fixing the .119 KL=0.17
mismatch?

**Verdict: NO. Adopt option (i) — vendor THRML's block-Gibbs directly.**

### Key findings

**(a) Structural mismatch at finite K confirmed.** The MH and Gibbs
transition kernels are mathematically distinct and do not commute, even
when targeting the same Boltzmann measure π:

- **Per-step TV**: MH accepts energy-lowering moves with probability 1;
  Gibbs (Heat-Bath) accepts with sigmoid probability `1 / (1 + e^{−Δ/t}) < 1`.
  Off-diagonal transition entries differ from K=1 onward.
- **Sample-mean energy bias**: MH is strictly greedier — deterministically
  accepts all downhill moves — and cascades into local minima faster than
  Gibbs under finite-time averaging (K ≤ 100). Generates divergent
  sample-mean energy bias.
- **Higher-order statistics**: Single-site MH suffers from rejection
  self-loops (stuttering), drastically increasing autocorrelation time
  versus block-Gibbs's parallel-coloring updates that randomize entire
  conditionally-independent vertex sets per sweep.

**(b) For n=128 fully-connected at β=1: K ≥ exp(Ω(n)).** The setup is a
dense unscaled Sherrington-Kirkpatrick (SK) spin glass operating deep
in the replica-symmetry-breaking glassy phase (T=1 ≪ T_c ≈ 6.5; local
field σ ≈ 6.5). For MH and block-Gibbs to align to KL < 0.05 requires
both chains to wash out their divergent transient dynamics — i.e.,
mix to stationarity. Mixing time is dominated by Arrhenius crossing of
O(n) energy barriers, giving **K ≫ 10¹⁵ sweeps** for parity at n=128.
**Computationally astronomical and infeasible.**

**(c) Algorithm 2 CANNOT recover block-Gibbs.** Two independent reasons:

1. **Graph-coloring collapse**: A fully-connected graph is the clique
   K_128. Any valid independent-vertex coloring uses exactly 128 colors,
   so maximum block size = 1. Block-Gibbs degenerates to single-site
   Gibbs, which (per (a)) still diverges from MH.
2. **Differentiability catch-22**: On sparse graphs where blocks > 1,
   recovering block-Gibbs requires the MH acceptance ratio to equal 1.0
   identically. But MCMC Layers' Fenchel-Young K=1 gradient
   (Proposition 3) is derived from the score function of the acceptance
   ratio. **If acceptance is fixed at 1.0, the Δ-gradient signal
   vanishes — destroying the differentiability premise of the paper.**
   Any other proposal yields Block-Metropolis, which structurally
   diverges from Block-Gibbs.

**(d) Minimum K dominated by spectral gap.** On the dense spin glass,
the spectral gap is exponentially small. Production-feasible K (≤100)
cannot wash out the transient kernel mismatch.

### Falsifiable empirical predicate (zero-coupling test)

Set `J_ij = 0`, `h_i = 0` (infinite temperature, ΔE = 0 for all moves).
Start both samplers from `y_0 = [-1, -1, ..., -1]`. Run K=1 sweep.
Measure expected Hamming distance from initial state.

| Sampler | K=1 behavior | Expected Hamming |
|---|---|---|
| Gibbs (THRML) | `1/(1+e^0) = 0.5` per spin → randomizes perfectly | 64 (binomial center) |
| MH (Carnot/MCMC Layers) | `min(1, e^0) = 1` accept all → deterministic invert OR unrejected random walk | 128 (systematic) or ≈55.3 (random scan) |

KL between perfect binomial and deterministic/Poisson-binomial at K=1
is mathematically massive. **This isolates the operator mismatch from
landscape effects and proves the divergence is baked into the kernels,
not a learnable mixing-time issue.**

### Action: vendor THRML directly

> "Because Carnot is an EBM verification framework auditing a reference
> simulator at K ≤ 100 ≪ τ_mix, your true mathematical target is not
> the asymptotic Boltzmann distribution π, but the specific
> non-equilibrium transient distribution `P_0 T_THRML^100`."

Carnot must execute THRML's exact Markov transition operator: independent-
set graph coloring, block-conditional heat-bath math, parallel scan
schedule, exact JAX PRNG key consumption paths. **Vendoring THRML
guarantees alignment with the reference target.**

### Implications for follow-up prompts (cascade)

- **DT-MCMC-K1** (Phase 5 in-situ training quality): RELEVANCE
  *narrowed*. MCMC Layers is no longer the inference-time sampler;
  it could still be a Phase 5 *training-time* component if the goal
  there is differentiable PCD on a non-THRML target. Consider rescoping
  the prompt to: "given that we vendor THRML for inference, can MCMC
  Layers' K=1 Fenchel-Young loss train the verifier-coupling parameters
  θ such that θ-induced THRML samples match data?"
- **DT-MCMC-STATELESS** (production deployment): NOW MOOT for the
  sampling role. THRML is library-shipped (no persistent chain
  required). Question may still be relevant for any auxiliary MCMC
  Layer use.
- **DT-COMPOSITION**: REVISED. Three-sampler composition becomes
  [THRML-vendored block-Gibbs] for (I) inference sampling,
  [Spectral Annealing] for (II) optimization,
  [BRAIN or MCMC Layers] for (III) distribution learning.

### Tasks to file

1. `.120 priority update`: drop the
   `exp15ZZ-iclr26-mcmc-layer-as-sampler-fix` task seed; replace with
   `exp15ZZ-thrml-vendored-block-gibbs-replacement`.
2. `.120 audit task`: Run the zero-coupling test (10 minutes of
   compute). Report Hamming-distance distributions for Carnot's Gibbs
   vs THRML at K=1. Confirms or refutes the operator-mismatch finding
   empirically before committing to vendoring.
3. **CLAUDE.md decentralization-rule check**: THRML 0.1.3 is Apache-2.0
   on PyPI. Per Rule 3 (distribution mirroring): we should fork THRML
   to a Carnot-controlled mirror (gitea + github) before depending on
   it as a load-bearing inference component, in case Extropic
   deprecates / re-licenses / removes it.

### Paper-v6 implications

- The KL=0.17 finding from .119 is **resolved cleanly** by vendoring
  THRML: Carnot's sampler IS THRML's sampler, parity is constructive.
- Paper-v6 §3 can claim "Carnot uses Extropic THRML 0.1.3 as the
  reference sampler implementation, vendored under Apache-2.0."
- The integration plan's Theme B (three orthogonal sampler options)
  collapses to two — Carnot's sampler is no longer a research question;
  it's THRML.

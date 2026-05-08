# Deep Think responses — ICLR 2026 follow-up

Live record of Deep Think's verdicts on the prompts drafted in
`iclr26-deep-think-prompts.md` + `iclr26-deep-think-prompts-batch2.md`.
Append responses as they arrive; each response should drive an
integration-plan update + .NNN priority adjustment.

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

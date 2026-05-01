# Phase 3a DBAE-EBM Pre-Prototype Threat Model

**Date:** 2026-05-01
**Status:** PRE-PROTOTYPE ADVERSARIAL ROUND — no code written yet
**Experiment:** exp1095
**Architecture under review:**

```
text → encoder (tanh → z ∈ [-1,1]^d) → EBM E(z) = -z^T J z → MCMC → z' → decoder → text'
Training: minimize E(z_correct) + max(0, margin - E(z_wrong))
```

This document applies the Phase Prototype + Empirical Validation + Adversarial Check Discipline
(CLAUDE.md MANDATORY) to the DBAE-EBM architecture *before* any prototype code is written.
The goal is to identify every way the prototype could silently pass acceptance gates without
actually learning the intended semantics.

**Pre-condition check (Phase 1c):** exp1093 result loaded. The current 3-verifier suite
(SpilledEnergyDetector, NUPProbeV4, PCIBProbe) has joint_null_space_fraction = 0.0 but
and_composition_viable = false because max pairwise r-correlation = 0.656 (above 0.5 threshold).
**Implication:** the training signal DBAE-EBM would receive from AND-composition of these three
verifiers is corrupted by correlated gradients — verifiers 1 and 2 see nearly the same
null space and reinforce each other's blind spots. This pre-condition is not met.

---

## Attack Pattern 1: Degenerate Identity Encoder

### How prototype can fail silently

The encoder maps text tokens to z via a linear layer followed by tanh. If the weight matrix
W happens to scale inputs to the saturation region of tanh (|Wx| >> 1), the encoder
effectively becomes a sign function: z ≈ sgn(Wx). In the other direction, if W → small
and b → 0, tanh(Wx + b) ≈ Wx (linear regime), effectively making z a scaled copy of
the input embedding — an identity-like pass-through.

Neither extreme compresses semantics. The identity path means: (a) the "bottleneck" carries
as many bits as the input embedding, (b) z has no alignment with the FPGA spin lattice, and
(c) the EBM is learning a surrogate that approximates the input embedding distance, not
linguistic constraint satisfaction.

The training loss will fall normally. L_recon improves because z contains enough information
to reconstruct text trivially. E(z_correct) will be lower than E(z_wrong) because correct
inputs map to different input-embedding regions than corrupted ones — the EBM is accidentally
learning an embedding-distance proxy, which correlates with the training labels but has
nothing to do with logical constraints.

### Why acceptance gate misses it

The acceptance gate checks (1) α_t survival, (2) bottleneck integrity via >85% joint-constraint
pass rate after 100 Langevin steps, and (3) transpilation: sgn(z) degradation <5%.

An identity encoder passes condition (3) trivially — if z already lives near ±1 due to
tanh saturation, sgn(z) = z and degradation is near 0. It passes condition (2) if the EBM
has learned a reasonable embedding-distance energy, which achieves >85% on training-set
verifiers. It passes condition (1) because α_t only tracks how many verified vs. self-generated
corrections are used — this is unaffected by whether the encoder compresses semantics.

### Required instrumentation

1. **Latent variance spectrum**: compute `var(z_i)` per latent dimension. An identity encoder
   will show all-high variance (near 1.0) without the bimodal ±1 saturation pattern that a
   good bounded encoder shows. Specifically: compute histogram of tanh pre-activations `u_i`.
   Identity: uniform distribution across all values. True compression: bimodal at ±large values.
2. **Singular value ratio of encoder weight matrix W**: compute SVD(W). Identity-like: singular
   values tightly clustered, ratio σ_1/σ_min ≈ 1. Good compression: wide spread indicating
   dimensions are selectively activated.
3. **Mutual information I(z; x) vs I(z; y_verifier)**: mutual information between z and raw
   input should be low (compressed bottleneck); mutual information between z and verifier
   labels should be high (semantic signal retained). An identity encoder flips this ratio.

### Minimum detection test

Train on 1000 examples. After warmup (Stage 1), compute:
- Mean |u_i| for i=1..d where u_i = arctanh(z_i) (pre-activation values)
- If mean |u_i| < 0.5 across >50% of dimensions → identity regime (W is too small)
- If all mean |u_i| > 3.0 → saturation regime (W is too large, acting as sign function)
- Target: bimodal distribution where most u_i ∈ [-3, -1] ∪ [1, 3]

---

## Attack Pattern 2: Decoder LM-Prior Overpowering Bottleneck

### How prototype can fail silently

A decoder trained on natural language implicitly learns the marginal token distribution of the
training corpus. If the latent dimension d is large enough, the decoder can learn a redundant
language-model prior in its parameters: it copies the most common token sequences regardless
of what z says, because minimizing L_recon on natural language with a large-capacity decoder
makes "always predict the most likely continuation" a locally optimal move.

This is the Bowman 2015 / OPTIMUS 2020 posterior collapse failure mode adapted to the
deterministic setting. The DBAE-EBM mitigates it by removing the Gaussian KL term, but
nothing in the architecture *prevents* the decoder from learning to ignore z if z → decoder
gradient is smaller than the decoder's intrinsic LM bias gradient.

Symptom: the decoder produces fluent text regardless of what value z takes. Varied z values
(including random draws) all decode to plausible-but-z-independent text. L_recon appears
excellent. The EBM's E(z_correct) is lower than E(z_wrong), but when z_wrong is perturbed
to a z_correct by the MCMC, the decoder outputs the same text as before the perturbation.
The "correction" signal is silently discarded.

### Why acceptance gate misses it

The >85% joint-constraint pass rate check (gate condition 2) is evaluated on the *original*
z values, not on perturbed z' from MCMC. If the decoder ignores z, it will still achieve
the pass rate because it's producing training-distribution text. The 100 Langevin steps
produces a z' that's "better" by the energy metric, but since the decoder ignores z, the
decoded text is unchanged — which may still pass 85% of verifiers on held-out data because
it's valid English.

### Required instrumentation

1. **Mutual information knockout test**: compute L_recon when z is replaced by a *random*
   draw from the empirical z distribution. If loss degrades by <5%, the decoder is ignoring z.
   Target: >20% degradation when z is randomized.
2. **DecodedTextDiversity across varied z**: use `DecodedTextDiversity` from
   `python/carnot/eval/diagnostics.py`. Draw 50 random z values. Decode each. Compute
   diversity score. If `is_degenerate(threshold=0.3)` returns True, decoder is producing
   near-identical text regardless of z — bottleneck is bypassed.
3. **z-perturbation response**: take a correct sample, compute z_correct. Apply random
   unit-norm perturbation of magnitude 0.5 to z_correct → z_perturbed. Compare decoded
   text. If text similarity (ROUGE-L) > 0.95 for 90% of perturbations, decoder ignores z.

### Minimum detection test

After Stage 1 (AE warmup), run the z-randomization test:
- Compute baseline L_recon on 100 held-out examples using true z.
- Reshuffle z values (assign z_i from example j to example k).
- Recompute L_recon with shuffled z.
- If shuffled_loss / baseline_loss < 1.1 (less than 10% degradation), the decoder is not
  using z and the bottleneck has failed.

---

## Attack Pattern 3: EBM Converging to Constants (J → 0)

### How prototype can fail silently

The EBM energy is E(z) = -z^T J z where J is a learned d×d coupling matrix. The contrastive
training signal is: lower E(z_correct), raise E(z_wrong). If J → 0, all energies become
equal (E(z) = 0 ∀z), and the training loss becomes:

  E(z_correct) + max(0, margin - E(z_wrong)) = 0 + max(0, margin - 0) = margin (constant)

This is a legitimate basin of the loss landscape. The optimizer may converge here because
it's a saddle point: decreasing J below zero accidentally makes "wrong" examples lower-energy
than "correct" ones, which the contrastive loss penalizes, pushing J back toward zero. The
optimizer oscillates near J ≈ 0.

Alternative: J has large diagonal but near-zero off-diagonal. E(z) = -Σ J_ii z_i^2 becomes
a per-dimension variance penalty, independent across dimensions. This is the "independent
feature" collapse — the EBM does not capture any joint linguistic structure, only per-feature
energies. Verification tasks that require multi-feature dependencies (e.g., subject-verb
agreement across a long context) will never be learned.

### Why acceptance gate misses it

The gate checks E(z_correct) < E(z_wrong) using the *margin-contrastive loss* — but if
the optimizer initializes near J ≈ 0, the training loop may show the loss decreasing toward
the constant `margin` value without J ever learning structure. The loss metric looks healthy.
The MCMC steps will wander randomly (flat energy landscape) but may still accidentally
produce z' values where the decoder outputs plausible text (because the decoder was trained
on a good AE in Stage 1).

An AUROC test on train/test set may show 0.5 (random) — but if test accuracy is reported
as "in range", this gets misinterpreted as "acceptable uncertainty" rather than "failed learning".

### Required instrumentation

1. **Frobenius norm of J over training**: track ||J||_F at each checkpoint. If ||J||_F < 0.01
   after 1000 steps, J has not escaped the origin — EBM is flat.
2. **Energy gap statistics**: for each batch, compute mean(E(z_wrong)) - mean(E(z_correct)).
   A healthy EBM should show positive and growing gap. Gap consistently near zero = J → 0.
3. **J spectral properties**: compute eigenvalues of J every 500 steps. A J that captures
   linguistic structure should have a few large eigenvalues (low-rank structure reflecting
   dominant constraint manifolds). Diagonal-dominated J = independent features only.
4. **AUROC of E(z)**: EBM energy should achieve AUROC > 0.65 distinguishing correct from
   wrong samples after 5000 steps. AUROC near 0.5 = J ≈ 0 failure.

### Minimum detection test

After Stage 2 (EBM warmup):
- Compute AUROC(E(z), label) on held-out 200 examples.
- Compute energy_gap = mean(E(z_wrong)) - mean(E(z_correct)).
- Gate: AUROC > 0.60 AND energy_gap > 0.1.
- If both fail → EBM has not learned a useful energy function. Do not proceed to Stage 3.

---

## Attack Pattern 4: Verifier Joint Null-Space Regression

### How prototype can fail silently

The DBAE-EBM uses verifier signals to shape the EBM energy during Stage 3 training. If the
k verifiers used share a large joint null space (a subspace of z where all verifiers return
the same score regardless of linguistic quality), the EBM learns to minimize energy in that
shared null space. The encoder is then implicitly incentivized to map inputs into the null
space — producing z values where the verifier ensemble cannot distinguish correct from wrong.

This is precisely what exp1093 measured for the Phase 1c verifier suite. Key finding:
- Joint null-space fraction = 0.0 (good on held-out data)
- BUT and_composition_viable = false because pairwise r-correlations are 0.41-0.66
- SpilledEnergyDetector ↔ NUPProbeV4 correlation = 0.656 (above 0.5 threshold)

The null space is not empty globally, but the verifiers are correlated: when one fails, the
other tends to as well. The EBM learns a manifold that satisfies "whatever SpilledEnergy
and NUP agree on" — which is a narrow criterion that misses adversarial examples lying in
their shared blind spots.

Phase 3 requires k=15 orthogonal verifiers for this to be safe. With only 3 correlated
verifiers available, the EBM will converge to a manifold that systematically ignores
adversarial patterns targeting the correlated null space.

### Why acceptance gate misses it

The acceptance gate checks >85% joint-constraint pass rate — but the gate verifiers are the
same verifiers used during training. The EBM has learned to minimize energy exactly on the
constraint manifold defined by these three verifiers. It will pass the gate at near-100%.
Novel adversarial examples targeting the shared null space of the correlated trio will
silently fail — but the gate doesn't include those.

### Required instrumentation

1. **Track r-correlations during training**: at each Stage 3 checkpoint, run NullSpaceEstimator
   from `python/carnot/eval/diagnostics.py` with the same verifier suite. If pairwise
   r-correlations *increase* during training (convergence to null space), the EBM is
   specializing away from the full constraint manifold.
2. **Adversarial null-space probing**: craft examples that are linguistically invalid but
   live in the joint null space of the verifier trio (i.e., examples where all three verifiers
   return scores near 0.5). Measure E(z) for these. If the EBM assigns low energy (correct-like)
   to null-space examples, the Phase 3 training signal is corrupted.
3. **Out-of-verifier-distribution accuracy**: hold out a fourth verifier not used during training
   (e.g., CausalReasoningVerifier from Tier 2.7). Measure constraint satisfaction on the decoded
   z' from MCMC. If the fourth verifier fails while the three training verifiers pass, the EBM
   has overfit to the training verifier manifold.

### Minimum detection test

Before Stage 3: measure pairwise r-correlations of all k verifiers. Gate:
- max pairwise r-correlation < 0.3 (well below exp1093's 0.656 finding)
- at least k=6 verifiers with diverse signal sources

If this gate fails: **do not launch Stage 3**. The training signal is already insufficient.
Note: exp1093's result confirms this gate will fail with the current 3-verifier suite.
Minimum action required before a valid DBAE-EBM prototype: add 3-5 additional verifiers with
distinct null spaces (e.g., add ThinkPRMProbe, SymCodeVerifier, CausalReasoningVerifier).

---

## Attack Pattern 5: Bottleneck Collapse via Gradient Vanishing (Tanh Saturation)

### How prototype can fail silently

In Stage 3 (asymmetric finetuning), the encoder receives two gradient streams:
(A) L_recon gradient: pushes encoder to preserve input information
(B) β · E(z_correct) gradient: pushes encoder toward lower-energy regions of J

For (B) to flow, the chain rule requires ∇_W E = ∇_z E · ∂z/∂W = ∇_z E · (1 - z²) · x^T.

The factor (1 - z²) is the tanh derivative. When z → ±1 (saturation, which L_var in Q1
regularization is designed to encourage), (1 - z²) → 0. The EBM gradient vanishes entirely.

The encoder trains fine on L_recon (which flows through the same tanh but is trained to
avoid saturation locally). But the EBM gradient is mathematically suppressed in the exact
regions where the encoder is most confident — which are the regions the FPGA needs to use
for spin readout (|z| near 1). The encoder saturates correctly, making transpilation feasible,
but the EBM cannot shape those saturated regions via backpropagation.

Result: the encoder's saturated dimensions are frozen to whatever the L_recon warmup produced.
The EBM only learns to shape the *unsaturated* intermediate dimensions. The FPGA spin lattice
(which reads sgn(z)) only interacts with the saturated dimensions. The EBM is shaping a
different subspace than the hardware will execute.

### Why acceptance gate misses it

The Transpilation Gap test (condition 3) checks whether sgn(z') degrades the constraint pass
rate by <5%. But if the EBM has only shaped the *unsaturated* z dimensions (which the FPGA
ignores), the gate will pass because both the continuous z' and sgn(z') use the same saturated
dimensions for text reconstruction. The degradation is near-zero — but for the wrong reason.
The FPGA sampler would need to flip *unsaturated* dimensions (which it can't, because it
only sees binary spins), so the hardware is effectively sampling from a flat energy.

### Required instrumentation

1. **Per-dimension tanh saturation tracking**: for each encoder output dimension i, track
   mean(|z_i|) and fraction of examples where |z_i| > 0.9. Dimensions with >90% saturation
   are "frozen" from the EBM's gradient perspective. Track over training — saturation should
   not monotonically increase once Stage 3 begins.
2. **EBM gradient magnitude by saturation level**: compute ||∂E/∂W_encoder|| separately for
   highly-saturated dimensions (|z| > 0.9) vs. mid-range dimensions (|z| < 0.5). If the
   highly-saturated dimensions receive <1% of the gradient norm, the EBM is not shaping the
   hardware-relevant subspace.
3. **Mirrored Langevin vs. direct z gradient comparison**: as described in the followup results,
   Mirrored Langevin Dynamics (MLD) evaluates energy in z-space and steps in unbounded u-space
   (u = arctanh(z)). This bypasses the (1 - z²) vanishing term. Verify MLD is implemented —
   not naive gradient descent in z-space — and confirm the gradient norms in u-space are not
   suppressed.

### Minimum detection test

After 100 Stage 3 training steps:
- Compute fraction_saturated = fraction of (example, dimension) pairs where |z_i| > 0.9
- Compute gradient_alive = ||∂E/∂W|| on saturated dimensions / ||∂E/∂W|| on all dimensions
- Gate: gradient_alive > 0.1 (EBM can shape at least 10% of the saturated subspace)
- If gradient_alive < 0.05 → Stage 3 EBM training is silent. Pivot to MLD (pre-activation
  space MCMC) as described in phase3-encoder-architecture-deep-think-followup-results.md Q3.

---

## Cross-Phase Dependency Analysis

### a. Cross-Phase Dependency Failure (Phase 1c pre-condition)

**Pre-condition:** DBAE-EBM Stage 3 requires verifiers with diverse null spaces for valid
training signal. The Phase 1c requirement is: pairwise r-correlation < 0.3 for all k verifier
pairs, with at least k=6 verifiers.

**exp1093 finding:** and_composition_viable = false. Pairwise r-correlations: 0.41-0.66.
Max correlation 0.656 exceeds the 0.3 threshold by 2.2×. Only 3 verifiers measured.

**Verdict:** Phase 1c pre-condition is NOT met. DBAE-EBM Stage 3 training on this verifier
suite will produce a corrupted training signal (Attack Pattern 4 above). Before any prototype
runs Stage 3, the verifier diversity problem must be resolved:
- Add ThinkPRMProbe (requires a GGUF model)
- Add SymCodeVerifier (Tier 2.5, already shipped)
- Add CausalReasoningVerifier (Tier 2.7, already shipped)
- Add HalluField (Tier 0e, already shipped)
- Re-run exp1093 with k=6+ verifiers; gate on max r-correlation < 0.3

Until this passes, DBAE-EBM Stage 3 should not run. Stages 1 and 2 can proceed independently.

### b. Decentralization Risk Assessment

**Rule 2 check:** does the DBAE-EBM prototype require a closed-weight model to function?

- Encoder: initialized from a pretrained backbone (e.g., RoBERTa). RoBERTa is Apache 2.0
  open-weight. Risk: NONE if RoBERTa-base is used; DEGRADED if initialized from a closed
  frontier model's encoder for embedding quality.
- EBM: E(z) = -z^T J z. Pure matrix multiplication. No external model dependency. Risk: NONE.
- Decoder: symmetric to encoder; same analysis. Risk: NONE with open-weight init.
- Verifiers used in Stage 3: ThinkPRMProbe requires a GGUF LLM. All current Tier 0 verifiers
  (SpilledEnergyDetector, NUPProbeV4, PCIBProbe) are training-free and local. Risk: LOW for
  basic prototype; MEDIUM if ThinkPRMProbe is required to meet Attack Pattern 4's diversity gate.

**Verdict:** DBAE-EBM prototype is decentralization-safe at the Stage 1/2 level. Stage 3
requires diverse verifiers — this can be satisfied entirely with local open-weight models
(GGUF Qwen3.6-35B for ThinkPRM, rule-based SymCodeVerifier, rule-based CausalReasoningVerifier).
No closed-weight dependency required. CLAUDE.md Rule 2 is satisfied.

**Required action:** ensure `cached_sota_pair()` pattern is used if any LLM-based verifier
is needed (experiment_template.py is the canonical path). ThinkPRM can use Qwen3.6-35B-A3B-GGUF.

### c. Hardware Portability Assessment

**Rule 5 check:** can the DBAE bottleneck run on KV260 / NPU?

**KV260 compatibility:**
- The continuous EBM E(z) = -z^T J z requires floating-point matrix multiplication on the
  continuous latent z ∈ [-1,1]^d. The KV260 has no floating-point hardware — it operates
  on 1-bit discrete spins. This is exactly the mismatch identified in the BONUS FLAG in
  phase3-encoder-architecture-deep-think-followup-results.md.
- The DAE-DEBM pivot (phase3-dae-debm-pivot-decision.md) resolves this by using
  z ∈ {-1,+1}^d via straight-through estimators, matching the hardware exactly.
- **DBAE-EBM as specified is NOT hardware-portable to KV260.** CLAUDE.md Rule 5 VIOLATED.

**NPU compatibility:**
- ONNX Runtime NPU execution providers (DirectML, OpenVINO, CoreML, Ryzen AI) can execute
  bounded matrix multiplications on integer or float16 tensors.
- The encoder tanh + linear layers and decoder sigmoid + linear layers are NPU-friendly.
- The EBM energy z^T J z is a quadratic form — expressible as two matmuls + elementwise ops.
- **DBAE-EBM is NPU-portable via ONNX.** CLAUDE.md Rule 5 SATISFIED for NPU path.

**Verdict:** DBAE-EBM is hardware-portable to NPU (sovereignty claim holds for consumer devices).
It is NOT hardware-portable to KV260 FPGA. For FPGA sovereignty, the DAE-DEBM pivot is required.
The position paper should anchor sovereignty claims to NPU, not FPGA, when describing DBAE-EBM.

---

## Instrumentation Checklist

The following diagnostics MUST be implemented in the DBAE-EBM prototype before any training run:

| ID | Diagnostic | Module | Gate |
|----|-----------|--------|------|
| D-01 | Tanh pre-activation histogram per dimension | `diagnostics.py` | Bimodal: most u_i ∈ [-3,-1]∪[1,3] |
| D-02 | SVD of encoder weight W (σ_1/σ_min ratio) | inline | Ratio ∈ [2, 50] — not identity, not degenerate |
| D-03 | z-randomization reconstruction loss ratio | inline | shuffled_loss/baseline_loss > 1.1 |
| D-04 | DecodedTextDiversity on random z draws | `diagnostics.DecodedTextDiversity` | Not degenerate (>0.3 diversity score) |
| D-05 | ||J||_F over training | inline | ||J||_F > 0.1 after 1000 steps |
| D-06 | Energy gap: mean(E_wrong) - mean(E_right) | inline | > 0.1 after Stage 2 warmup |
| D-07 | AUROC of E(z) on held-out set | inline (uses `auroc_batch`) | > 0.60 after Stage 2 |
| D-08 | NullSpaceEstimator pairwise r-correlations | `diagnostics.NullSpaceEstimator` | All pairs < 0.3 |
| D-09 | EBM gradient norm on saturated vs. all dims | inline | gradient_alive > 0.10 |
| D-10 | AlphaT tracker during Stage 3 MCMC | `diagnostics.AlphaT` | inf_t α_t > 0.1 |

All 10 diagnostics must be wired before the Stage 1 training loop begins. Any gate failure
stops the training run and writes a blocked artifact — do not proceed to the next stage.

---

## Summary: Acceptance Gates Per Stage

| Stage | Gates (all must pass) | Failure action |
|-------|-----------------------|----------------|
| Pre-Stage-1 | D-08 (verifier diversity): max r-corr < 0.3, k ≥ 6 verifiers | Resolve Attack Pattern 4 first |
| Post-Stage-1 | D-01 (bimodal pre-activation), D-02 (SVD ratio), D-03 (z-randomization) | Block Stage 2 |
| Post-Stage-2 | D-05 (||J||_F > 0.1), D-06 (energy gap > 0.1), D-07 (AUROC > 0.60) | Retrain EBM; check J initialization |
| Pre-Stage-3 | D-04 (diversity), D-09 (gradient alive > 0.10) | Switch to MLD if gradient vanishing |
| Post-Stage-3 | D-10 (α_t > 0.1), joint-constraint pass rate > 85%, sgn(z) degradation < 5% | Reject architecture |

Note: the sgn(z) degradation < 5% gate is only valid if the DAE-DEBM pivot has NOT been taken.
If using discrete latent (z ∈ {-1,+1}^d via STE), this test is trivially satisfied by construction.
The more meaningful gate for discrete is the manifold dead-zone test (50 Glauber steps → decode
→ check output is semantically valid, not gibberish).

---

## Architectural Recommendation

Based on the five attack patterns above, the adversarial reviewer recommends:

1. **Do not build the continuous DBAE-EBM as the primary prototype.** The tanh saturation +
   gradient vanishing interaction (Attack Pattern 5) and the hardware mismatch (KV260
   incompatibility) are fundamental. The DAE-DEBM pivot (phase3-dae-debm-pivot-decision.md)
   resolves both.

2. **The continuous DBAE-EBM is valid as an NPU-targeted variant** (sovereignty via consumer
   hardware, not FPGA). Build it as the NPU branch, not the primary architecture.

3. **Block Stage 3 until exp1093's pre-condition is met.** This requires measuring 6+ verifiers
   with max pairwise r-correlation < 0.3. The current 3-verifier suite fails this gate.

4. **Wire all 10 diagnostics (D-01 through D-10) before any training run.** These are not optional
   monitoring — they are the gates that prevent each attack pattern from passing silently.

5. **The Mirrored Langevin Dynamics (MLD) implementation from the followup Deep Think (Q3) is
   mandatory** for the continuous variant — naive gradient steps in z-space exhibit the tanh
   gradient vanishing described in Attack Pattern 5. Use u = arctanh(z) as the unconstrained
   parameterization for all MCMC steps.

---

Cross-references:
- Architecture: `docs/research-notes/phase3-encoder-architecture-deep-think-results.md`
- Followup: `docs/research-notes/phase3-encoder-architecture-deep-think-followup-results.md`
- DAE-DEBM pivot: `docs/research-notes/phase3-dae-debm-pivot-decision.md`
- Phase-3 blind spot audit: `docs/research-notes/phase3-architecture-blindspot-audit-results.md`
- Pre-condition (Phase 1c): `results/experiment_1093_phase1c_verifier_joint_null_space_measurement.json`
- Diagnostics module: `python/carnot/eval/diagnostics.py`

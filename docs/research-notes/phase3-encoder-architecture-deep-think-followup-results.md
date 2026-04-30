# Phase-3 Encoder Architecture — Deep Think Follow-up Results

**Date:** 2026-04-30
**Prompt:** `phase3-encoder-architecture-deep-think-followup-prompt.md`

## Q1 — Encoder regularization without ELBO

**Recommendation: VICReg/Barlow Twins variance + decorrelation +
masked token reconstruction.**

```python
# Force binarization by pushing variance to theoretical max (1.0)
L_var = lambda_1 * torch.mean(F.relu(1.0 - torch.var(z, dim=0)))
# Penalize off-diagonal covariance to force orthogonal utilization
L_cov = lambda_2 * torch.sum(torch.triu(torch.cov(z.T), diagonal=1)**2)
L_reg = L_var + L_cov
```

**Why this is principled:**
- **The Binarization Trick** — for any variable strictly bounded in
  `[-1, 1]^d`, the maximum theoretical variance is 1.0, achieved
  *if and only if* the distribution is Rademacher (mass saturated
  at ±1). Penalizing variance below 1.0 mathematically eliminates
  origin-collapse, forces full latent utilization, AND primes for
  Phase-2 `sgn(z)` transpilation — without any variational prior.
- **Masked token reconstruction** (30-50% drop) prevents the decoder
  from ignoring `z` and relying on local autoregressive context.

## Q2 — Joint loss formulation

**Recommendation: Three-stage training. Naive single-loss joint is
fatally unstable.**

The core insight: backpropagating Contrastive Divergence
negative-phase gradients into a deterministic encoder will **collapse
the input manifold to a single point** to trivially minimize EBM
energy, destroying reconstruction. This is the failure mode that
kills the obvious `L = L_recon + β · L_energy` formulation.

| Stage | Trains | Loss |
|---|---|---|
| 1. Manifold Warmup | AE only | `L_recon + L_reg` (Q1's regularization) |
| 2. EBM Warmup | EBM (AE frozen) | Standard CD on fixed `z` manifold |
| 3. Asymmetric Finetuning | Both unfrozen | EBM: CD; Encoder: `L_recon + β · E(z_real)` with **stop-gradient on EBM's `z_fake` MCMC gradients** |

Start with `β` so small that `||∇L_energy|| ≈ 0.01 · ||∇L_recon||`.
The verifier gently shapes the encoder toward valid geometries while
the deterministic bridge stays preserved.

## Q3 — Langevin MCMC in tanh-bounded space

**Recommendation: Mirrored Langevin Dynamics (MLD).**

If we define `E(u)` on unbounded pre-activations `u` and threshold
later, we hit fatal gradient starvation: `∇_u E = ∇_z E · (1 - tanh²(u))`
vanishes at boundaries — exactly where useful spins live.

MLD solution: evaluate energy in bounded space, take steps in the
unbounded **dual** space via a mirror map (`arctanh` for the box):

```
u_{t+1} = arctanh(z_t) - η ∇_z E(z_t) + sqrt(2η) ε
z_{t+1} = tanh(u_{t+1})
```

This **provably converges** to `p(z) ∝ exp(-E(z))` restricted to the
box, **bypasses** the `1 - tanh²` vanishing term, and **avoids**
reflective-boundary stalling.

## 🚨 BONUS FLAG — Reconsider Candidate D

**Deep Think raised a critical architectural concern:**

> If your Phase-2 FPGA specifically executes discrete Glauber
> spin-flips on a ±1 lattice, a continuous EBM is structurally
> mismatched. The continuous EBM will learn interpolating gradient
> paths that physically do not exist in hardware.

**Pivot recommendation:** Discrete Autoencoder (straight-through
estimators for `z ∈ {-1, 1}^d`) + natively Discrete EBM trained via
exact Gibbs/Glauber sampling. **Take the regret now to guarantee
zero transpilation degradation.**

This is conditional on what KV260's `ising_sampler_v2.v` actually
executes:

| Hardware primitive | Architecture choice |
|---|---|
| Continuous gradient descent on continuous spin variables | Stay with DBAE-EBM + MLD |
| Mixed (continuous gradient + periodic spin readout) | Hybrid; DBAE-EBM may still work |
| **Pure discrete Glauber spin-flips** (likely the case) | **Pivot to Discrete AE + Discrete EBM** |

**Action item:** before scaling, confirm the KV260 hardware primitive
by reading exp1081's `ising_sampler_v2.v` and the Vivado synthesis
config. If discrete Glauber, pivot.

## Q4 — Smallest meaningful pre-1B-scale validation

**Recommendation: 100M params + FoVer 6,500-pair corpus + 10K steps,
initialized from a pretrained backbone (e.g., RoBERTa).**

This scale **cannot** evaluate emergent reasoning generalization, but
**can** definitively isolate architectural physics. Two structural
go/no-go tests:

### Test 1 — Transpilation Gap (Fast Fail)

Evaluate `L_recon` on training set using continuous `z`. Substitute
`z_bin = sgn(z)`. **If reconstruction exact-match degrades >5%, the
continuous bottleneck cannot support hardware quantization.**

### Test 2 — Manifold Dead-Zone Test

A DBAE lacks a VAE's KL prior, so it provides **no mathematical
guarantee of a smoothly interpolable latent space**. Run 50 MLD steps
starting from a valid `z`. Pass `z'` to the decoder.

**If the decoder outputs unrecoverable gibberish**, the continuous
EBM is successfully finding low energy, but doing so in undefined
"dead zones" between isolated data clusters.

If either test fails, a 1B-scale run will unconditionally fail.

## Implications

### Three concrete decisions before the prototype

1. **Confirm KV260 hardware primitive** — read
   `ising_sampler_v2.v` from exp1081's deliverable. If discrete
   Glauber, pivot to Discrete AE + Discrete EBM.
2. **Pick the regularization coefficient pair** (λ₁, λ₂) — start
   with `λ₁ = λ₂ = 1.0` per the Barlow Twins original recipe; tune
   on the 6,500-pair corpus.
3. **Implement MLD or Glauber** — depends on (1).

### Two non-negotiable tests pre-scale

- Transpilation Gap (`<5%` degradation under `sgn(z)`)
- Manifold Dead-Zone (decoder output stays meaningful after 50 MLD
  steps)

If either fails on the small prototype, the 1B-scale run is
mathematically doomed.

### Architecture matrix (post-Round 2)

| Path | When | Loss formulation | MCMC |
|---|---|---|---|
| **DBAE-EBM** (continuous) | KV260 supports continuous gradient | 3-stage warmup, asymmetric finetune, stop-grad on z_fake | Mirrored Langevin |
| **DAE-EBM** (discrete pivot) | KV260 only does discrete Glauber | Same 3-stage, but with straight-through estimators | Native Gibbs/Glauber |

The pivot is conservative: it eliminates one failure mode
(continuous-discrete mismatch) at the cost of straight-through
gradient noise.

## Cross-references

- Original DBAE-EBM result:
  `phase3-encoder-architecture-deep-think-results.md`
- exp1081 FPGA scale benchmark (provides the
  `ising_sampler_v2.v` source to verify hardware primitive)
- `project_zenil_alpha_grounding.md` (α_t threshold for the gate)
- `project_continuous_ising_rank.md` (Phase 2 transpiler)

# Phase-3 DAE-DEBM Round 3 — Deep Think Result

**Date:** 2026-04-30
**Prompt:** `phase3-dae-debm-round3-deep-think-prompt.md`

## TL;DR — Pivot-back to DBAE-EBM with deployment-time distillation

The Round 2 → Round 2.5 pivot to DAE-DEBM was **premature**. Round 3's
bonus flag identifies a previously-overlooked hardware constraint that
makes BOTH the discrete-EBM and continuous-EBM approaches fail to run
natively on the KV260 — but for different reasons.

The FPGA's sigmoid-LUT + LFSR Glauber sampler is hardware-optimized
not just for `±1` spins, but specifically for **quadratic Ising
energies** `E(z) = z^T J z + h^T z`. The flip probability formula is
`P = σ(-2β·ΔE_i)` where `ΔE_i` is computed as a sparse matrix dot
product against neighbors. **A deep-NN EBM cannot run natively** —
each `ΔE_i` would require a full forward pass.

Both pivot candidates fail this constraint:
- **DAE-DEBM (discrete deep EBM):** the discrete-EBM is still a deep
  NN, still needs forward passes, still can't run on the quadratic
  FPGA primitive.
- **Quadratic-only EBM:** `O(d²)` parameters can't capture the
  AND-composition of k=15 complex language verifiers.

**Resolution:** train a deep continuous EBM (DBAE-EBM with MLD), then
**distill at deployment** to a quadratic `(J, h)` Ising matrix via
Taylor expansion. The FPGA executes the distilled approximation; the
training loop benefits from full deep-NN expressivity.

This means the architecture is back to **DBAE-EBM** as Round 1
originally chose, with the addition of an explicit
**EBM-to-Ising distillation step** at deployment.

## Q1 — Hard tanh STE *(applied to continuous DBAE; was for DAE)*

```python
z = torch.sign(logits).detach() + logits.clamp(-1, 1) - logits.clamp(-1, 1).detach()
```

Why hard tanh wins:
- **Identity STE** causes logit runaway → encoder logits saturate to
  ±∞ to artificially maximize energy margins → permanent dead
  encoder neurons.
- **Gumbel-Softmax / Concrete** reintroduces continuous relaxation
  during training; on a hardware that enforces `±1`, this resurrects
  the very transpilation gap we tried to avoid.
- **Hard tanh** clips gradients for confident spins (`|logits| ≥ 1`),
  acts as a structural regularizer routing gradients to "undecided"
  spins near the decision boundary.
- Note: scale backward pass by `1/√d` for `d ≥ 256` to prevent
  RoBERTa encoder destabilization.

**Updated context:** with the pivot-back to continuous DBAE,
hard tanh isn't strictly required (the latent is already continuous).
But the same gradient-clipping intuition applies to the
deep EBM's input layer — wrap it with the same trick when training.

## Q2 — PCD with ε-greedy replay buffer

- NCE on text: defining a noise distribution that shadows the sparse
  `2^d` text manifold is unsolved. Classifier separates trivially,
  no gradient signal.
- Exact Gibbs: combinatorial blowup beyond `d ≈ 20`.
- Standard PCD: catastrophic mode collapse in discrete spaces; chains
  freeze in high-energy traps.

**Recipe:** PCD with persistent replay buffer of 10,000 `z`-states.
Each gradient step: 95% chains from buffer, 5% from random uniform.
Inject 5% Hamming noise (random bit-flips) into buffer-fetched chains
before applying Glauber steps. Random restarts + noise injection are
mandatory to escape Hamming dead zones.

**Updated context:** with the pivot-back, training EBM is on
continuous `z ∈ [-1,1]^d` via Mirrored Langevin. The replay-buffer +
noise-injection pattern still applies — substitute "Gaussian noise
on z" for "Hamming bit-flips."

## Q3 — d ∈ {256, 512}, structural floor d > m + k

The minimum viable `d` is bounded below by the language manifold's
intrinsic dimensionality `m` plus the AND-composition rank `k`. Each
verifier roughly halves the valid state space (consumes ~1 bit of
representational capacity).

For a 32-token text chunk: `m ≈ 128 bits`. With `k = 15` verifiers:
mathematical floor `d ≥ m + k ≈ 144`. `d ∈ {256, 512}` provides
massive combinatorial headroom (`2^512 ≈ 10^154`) without bottling
the FPGA clock cycle for distilled inference.

`d ≤ 64` causes severe expressivity collapse — distinct semantic
concepts forced to collide on the same latents → high decoder
perplexity. `d ≥ 1024` makes Hamming traversal kinetically
intractable even for parallel sweeps.

**Updated context:** the same floor applies to continuous DBAE
(intrinsic-entropy bound is the same; the verifier composition rank
is the same). Sweep `d ∈ {256, 512}` regardless of discrete/continuous
choice.

## Q4 — Parallel Tempering at training, NOT hardware-matching anneal

**Don't** replicate the hardware's transient hot-to-cold annealing
during EBM training. Maximum likelihood requires negative samples
drawn from the model's *equilibrium* distribution at the target
`β=1`. PCD over an annealing trajectory violates detailed balance,
biases gradients away from the true `Σ E_i(z)` verifier minima.

But static `β=1` PCD freezes chains in dead zones.

**Solution: Parallel Tempering.** Maintain parallel chains at fixed
tiered temperatures `β ∈ {0.1, 0.4, 0.7, 1.0}`. Standard Glauber
updates per chain, periodic probabilistic state swaps between
adjacent chains. Hot chains explore; cold chains refine. **Evaluate
EBM loss strictly on the `β=1` chain.** The `inf_t α_t > 0.1` gate
applies to the equilibrium state.

**Updated context:** Parallel Tempering also applies to continuous
EBM training with MLD — replace "Glauber updates" with "MLD steps,"
keep the cross-temperature swap structure. The bias-from-non-
equilibrium argument is the same.

## Q5 — 500-1000 sweeps minimum + Denoising-AE training

The 50-step Manifold Dead-Zone test from Round 2 is a continuous-
space relic. In `R^d`, Langevin glides effortlessly through smooth
interpolations; in `{-1,+1}^d` (or even bounded `[-1,1]^d`), traversal
between distant valid concepts requires a full thermal cycle: heat to
escape current mode, random-walk through "meaningless text" dead
zones, cool to snap into next valid mode.

**Two changes:**

1. **Test criterion: 500-1000 sweeps**, not 50. Realistic for
   discrete; safer for continuous too.
2. **Train as a Denoising AE.** Inject 5-10% random bit-flips (or
   Gaussian noise in continuous) into `z_data` before decoder. This
   expands valid textual modes from isolated points into dense
   neighborhoods, giving Glauber/MLD safe landing zones in dead-zone
   intermediate states.

The Denoising-AE addition is the single most important Round 3
practical insight — it changes the prototype from "fragile" to
"likely-functional" by construction.

## Bonus — CRITICAL ARCHITECTURAL FLAG: pivot-back to DBAE-EBM

> Carnot's `ising_sampler_v2.v` uses sigmoid-LUT and LFSR to compute
> spin-flips via `P = σ(-2β · ΔE)`. **This is hardware-optimized
> strictly for quadratic Ising energies** `E = z^T J z + h^T z`,
> where `ΔE_i` is computable via a highly parallel, sparse matrix
> dot-product.
>
> If your Latent EBM is a deep neural network, computing a single
> `ΔE_i` requires a full forward pass, which the FPGA physically
> cannot execute natively. Your EBM is therefore structurally
> restricted to an `O(d²)` quadratic matrix, which inherently lacks
> the deep capacity to capture the AND-composition of `k=15` complex
> language verifiers.

**The pivot to DAE-DEBM was misdiagnosed.** Discrete latent doesn't
fix the deeper problem: the FPGA isn't a general discrete sampler,
it's a *quadratic-Ising specifically* sampler. Both DAE-DEBM and
DBAE-EBM with deep neural EBM hit the same wall — neither can run
deep-NN-energy on the FPGA primitive.

### Resolution

**Train: DBAE-EBM with full deep-NN energy** (continuous bounded
latent, MLD, Round 2's 3-stage training, Parallel Tempering, all of
Round 3's STE / replay-buffer / Denoising-AE refinements).

**Deploy: distill the deep continuous EBM into a quadratic Ising
`(J, h)` matrix via Taylor expansion** around an anchor state.
The FPGA executes the distilled quadratic approximation. The
deep-NN expressivity is preserved at training time (where it
matters for k=15 verifier composition), and the deployment-time
quadratic restriction is paid via the distillation residual.

Distillation options:
- **Local Taylor expansion** around mean of `z`: keep up to 2nd
  order, map `½ z^T H z + linear` → Ising `J = -H/2, h = -∇E`.
  Approximation error `O(||z - z_0||³)` — works if FPGA inference
  stays near anchor.
- **Global least-squares fit** of quadratic `E_q(z)` against
  samples `(z, E(z))` from the deep EBM. Less interpretable but
  more globally accurate.

The "transpilation gap" is now non-zero by definition (it's the
distillation residual). But it's a known, bounded engineering cost
rather than a structural model-vs-hardware mismatch.

## Updated architecture decision

| Phase | Architecture | Detail |
|---|---|---|
| **Training** | **DBAE-EBM with MLD** | Continuous bounded latent `z ∈ [-1,1]^d`, deep-NN energy, Mirrored Langevin Dynamics, Parallel Tempering, hard-tanh STE on encoder pre-activations, Denoising-AE bit-flip noise on `z_data` before decoder, Round 2's 3-stage training. |
| **Deployment to FPGA** | **Distillation to quadratic Ising** | Taylor expand or least-squares fit `E_deep(z) → E_quad(z) = z^T J z + h^T z`. Deploy `(J, h)` to KV260. |
| **Deployment to GPU** | Run deep continuous EBM directly | No distillation needed; GPU has the forward-pass capacity. |

This means **the FPGA-vs-GPU baseline experiment** I scoped for .85
becomes more interesting — the FPGA runs a *distilled approximation*
of the GPU's deep EBM. Compare:
- GPU: deep continuous EBM, MLD, full expressivity
- FPGA: distilled quadratic Ising, faster but lower-fidelity
- Quality vs latency trade-off is the actual measurement

## Implications

### Phase-3 prototype plan changes

The prototype now has **three components**, not two:
1. Deep continuous EBM training (DBAE-EBM with MLD)
2. **Distillation step**: deep `E_deep(z)` → quadratic `(J, h)`
   for FPGA deployment
3. Deployment-time evaluation: how much accuracy is lost in
   distillation?

The Round 2 acceptance gate gets a fourth condition:
- `inf_t α_t > 0.1` (continuous trajectory) — measured on deep EBM
- Decoder >85% joint-constraint pass rate — measured on deep EBM
- `sgn(z')` binarization <5% degradation — measured on deep EBM
  output before distillation
- **NEW: distillation residual <X%** — measured between deep EBM
  predictions and quadratic-Ising approximation. The exact threshold
  depends on what fidelity Phase-2 needs.

### Position paper updates

Phase-3 section:
- DBAE-EBM is the chosen architecture (Round 1 was right)
- Round 2's "discrete pivot" was a useful checkpoint but didn't
  account for the quadratic-only FPGA constraint
- Distillation-to-Ising is the bridge; this is novel and worth
  highlighting
- Cite all three Deep Think rounds + the hardware verification

Phase-2 section:
- exp1081's FPGA scale benchmark stands, but the framing changes:
  the FPGA isn't running "the EBM" — it's running a distilled
  quadratic approximation. Performance numbers measure the Ising
  primitive, fidelity numbers measure the distillation gap.

### Memory + research notes

- Update `project_dbae_ebm_phase3.md` to note the pivot-back
- Mark `phase3-dae-debm-pivot-decision.md` as superseded
- Add this Round 3 result as the canonical decision

## Open questions for a Round 4 (if needed)

1. **Distillation method choice.** Local Taylor vs global
   least-squares vs neural-architecture-search for the best
   quadratic surrogate? Are there published works on neural-EBM
   → Ising distillation that we should reference?
2. **Fidelity threshold.** What's the maximum acceptable
   distillation residual for Phase-2 to count as "deployable"?
3. **Hybrid GPU+FPGA inference.** Could the GPU run the deep EBM
   for the first few MCMC steps (fast convergence), then hand off
   to FPGA for fast quadratic refinement? This is a fundamentally
   new deployment topology.

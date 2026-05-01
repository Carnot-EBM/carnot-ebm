# Phase-3 Encoder Architecture — Deep Think Follow-up Prompt

**Status:** Drafted 2026-04-30 after the original DBAE-EBM round.
Ready for paste when Deep Think quota resets.

**Why follow up:** the original round picked the architecture but left
four load-bearing implementation questions unspecified. Without
answers, we can't ship a small-scale prototype, so we can't validate
before scaling.

---

## Prompt to paste

You previously chose Candidate D (Deterministic Bounded Autoencoder +
Latent EBM, DBAE-EBM) for Carnot's Phase-3 Kona-parity foundation
model:

```
text → deterministic encoder → bounded z (tanh in [-1, 1]^d)
     → EBM(z) MCMC → z' → deterministic decoder → text'
```

Four implementation questions remain that we cannot resolve from
first principles. Each blocks small-scale prototype work.

### Question 1 — Encoder regularization without ELBO

By dropping the variational Gaussian prior + KL, you eliminate
posterior collapse. But you also eliminate the only force that
prevents **degenerate encodings**: a deterministic encoder is free
to (a) compress all signal into a tiny subspace of `[-1, 1]^d`,
leaving most dimensions saturated at the tanh boundary, or (b) learn
an effectively-constant embedding that the decoder ignores while a
strong language model in the decoder reproduces the input.

VAE's KL forced full latent utilization. What's the analogous
forcing function for the deterministic AE?

Candidates we've considered:
- L2 regularization on `z` activations (pulls toward origin, may
  collapse signal)
- Decorrelation penalty (`||cov(z) - I||_F`)
- Reconstruction with masked tokens (forces `z` to carry context not
  in nearby tokens)
- Adversarial discriminator on `z` (matches a target distribution
  without explicit prior)
- Information-theoretic regularization (e.g., `I(x; z)` lower bound
  via InfoNCE or Barber-Agakov)

Which of these — or a different mechanism — is the principled choice
that preserves DBAE's structural advantages (no collapse, clean
sgn() transpilation, deterministic gradient) while preventing the
degenerate-encoding failure mode?

### Question 2 — Joint loss formulation

The training corpus has two competing objectives:

- **Reconstruction**: `||decoder(encoder(x)) - x||` — the AE must
  preserve enough information to round-trip.
- **Verifier energy**: `E(z)` should be low for valid `x` and high
  for invalid `x` — the EBM must rank by constraint satisfaction.

These can fight: a `z` that minimizes reconstruction may not be the
`z` that makes the verifier signal monotone. Standard VAE-EBM
hybrids combine `L_recon + α · L_KL + β · L_energy` and tune
α/β. What's the analogous combination for DBAE-EBM that preserves
the *deterministic-bridge / energy-gradient-survival* property
the original round emphasized?

Possible answers:
- `L = L_recon + β · L_energy(encoder(x))` (single-loss two-term)
- Two-stage: train AE first to convergence, freeze encoder, train EBM
  in latent (may decouple verifier signal from encoder)
- Joint with curriculum: start `β = 0` (pure recon), ramp up over
  training (avoids collapsing to verifier-ignoring AE)
- Gradient-balancing schemes (PCGrad, GradNorm)
- A different formulation entirely

If the answer involves a hyperparameter, what's a defensible
starting value or a principle for picking one?

### Question 3 — Langevin MCMC mixing in tanh-bounded space

The bounded latent `[-1, 1]^d` is created by tanh squashing of
unbounded encoder pre-activations. Two failure modes for MCMC there:

(a) **Boundary saturation kills gradients.** Near `|z| ≈ 1`, the tanh
derivative `1 - tanh²(·)` is near zero. If the EBM minimum is at
the boundary (which it often will be when `sgn(z')` is the binary
spin we care about), Langevin steps `z ← z - η·∇E(z) + √(2η)·ε`
become arbitrarily small there. Mixing time blows up.

(b) **Reflective vs absorbing vs unbounded sampling.** Langevin in
unbounded ℝᵈ has well-known mixing rates. Bounded domains with
reflective boundaries are studied less. Does the MCMC even converge
to the correct stationary distribution under tanh-bounded dynamics?

What's the recommended Langevin variant for `[-1, 1]^d` that:
- Mixes acceptably at the boundary (where the most useful spins
  live, given the `sgn(z')` Phase-2 transpilation step)
- Provably converges to `p(z) ∝ exp(-E(z))` restricted to the box
- Is compatible with the FPGA's actual sampling primitive (which is
  a Glauber-style spin-flip on `±1` lattices, not gradient Langevin)

Specifically: should the EBM operate in *unbounded* space and only
threshold-to-`[-1, 1]^d` for hardware deployment? Or does the
training-time bounded constraint genuinely help the architecture?

### Question 4 — Smallest meaningful pre-1B-scale validation

The acceptance gate you specified is 1B-token scale. We can't justify
that compute spend without a pre-validation that DBAE-EBM works at
all. What's the **smallest tractable scale** that produces a
meaningful go/no-go signal?

Concretely, Carnot's current FoVer corpus is ~6,500 reasoning step
pairs (exp1055/1077). Could a 10K-step DBAE-EBM prototype on this
corpus already demonstrate:

- α_t survival on the small trajectory (with appropriately scaled
  threshold)?
- Sufficient reconstruction quality that the decoder doesn't crush
  signal?
- Non-trivial `sgn(z')` Boolean transpilation degradation
  (>5% would be a fast fail; <5% would be encouraging but not
  conclusive)?

If yes — what's the corpus / parameter / step count combination that
gives the highest signal-to-cost ratio for a Phase-3 small-scale
proof-of-concept?

If no — what's the minimum scale at which DBAE-EBM's structural
advantages over VAE-EBM and Diffusion-EBM are *separable* from
training-noise artifacts?

## Output format

Four numbered answers, each ≤ 250 words, with a concrete
recommendation we can implement in code. If any question doesn't have
a clean answer, say so explicitly and identify what we'd need to know
to answer it (more empirical data, a specific theoretical result,
etc.).

Bonus: if any answer suggests the original Candidate D pick should be
reconsidered (e.g., "the deterministic AE's regularization story is
fundamentally weaker than VAE's, here's the reframing"), flag it
explicitly. We'd rather take the regret early than at 1B-scale.

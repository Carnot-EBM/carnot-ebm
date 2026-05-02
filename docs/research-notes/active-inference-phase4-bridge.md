# Active Inference Phase-4 Architectural Bridge

**Status:** Committed research direction (2026-05-02 user directive).
**Sharpened 2026-05-02 07:10Z** based on the "Continuous vs Discrete
Paradigms" deep-research document: this is NOT a paradigm pivot, it
is an **inference-mode extension** of Carnot's existing Phase 1+3
infrastructure.

The original framing (Phase 4 as a "separate active-inference agent
built on PyMC") was a category error. The mathematical equivalence
`score(x) = ∇_x log p(x) = -∇_x E(x)` means Carnot's existing energy
gradient IS the score function for both HMC sampling AND diffusion-
of-thought refinement. We don't need a separate paradigm — we need
a different sampler on top of the existing `∇E`.

**Date drafted:** 2026-05-02
**Date sharpened:** 2026-05-02 07:10Z
**Sources:**
- Themesis Seed IQ public demonstration on ARC-AGI-3
  (0.95 demo + 1.00 reported leaderboard, 115% human action-efficiency)
- EBT/ARC-AGI-3 document positioning Carnot-EBM as paradigm exemplar
- "Continuous vs Discrete Paradigms" deep-research document
  (the score=energy-gradient equivalence)

---

## The hypothesis (sharpened 2026-05-02 07:10Z)

> **Carnot's existing energy landscape — defined by the k=N AND-
> composed verifier ensemble plus the DBAE-EBM — provides the
> exact `∇E` gradient required for both HMC sampling and diffusion-
> of-thought iterative refinement. Phase 4 swaps the sampler, not
> the paradigm.**

The mathematical bridge is exact, not approximate:

```
score(x) = ∇_x log p(x) = -∇_x E(x)   [for Boltzmann-distributed EBM]
```

A diffusion model's reverse denoising process uses `score(x)` to
move samples toward high-probability regions. HMC uses the same
`∇_x log p(x)` as the "force" pushing the system toward higher
probability. Carnot's energy function provides this gradient natively.

Therefore:

- **Phase 1 today** (Langevin/Gibbs sampler on `∇E`) is one
  point in the family of `∇E`-using inference algorithms.
- **Diffusion of Thought** is the same `∇E` with a different
  sampling schedule (forward noise → reverse denoise).
- **HMC sampling** is the same `∇E` with momentum-augmented
  dynamics (avoids random-walk inefficiency).

Phase 4 tests whether HMC + DoT-style sampling on Carnot's
existing `∇E` reproduces Seed IQ's claimed action-efficiency on
ARC-AGI-3, which would empirically validate the equivalence.

## Mathematical bridge

### Free Energy Principle (FEP) — variational free energy

Friston defines variational free energy as:

```
F[q] = E_q[log q(z)] - E_q[log p(x, z)]
     = D_KL[q(z) || p(z|x)] - log p(x)
```

where:
- `x` is observed data
- `z` is latent state (beliefs)
- `q(z)` is the agent's variational posterior
- `p(x, z)` is the joint generative model

The agent minimizes `F` by either:
1. **Perception** — updating `q(z)` to better match `p(z|x)` given fixed observations
2. **Action** — choosing actions that lead to observations consistent with `q(z)`

This is mathematically gradient descent on `F` with respect to `q` and action parameters.

### Carnot's verify-repair as `F` minimization

Carnot's k=N AND-composed verifier ensemble produces a per-candidate
score `s ∈ [0, 1]` where 1.0 = pass, 0.0 = violation. Define the
energy `E(x, y) = -log s(x, y)` for candidate output `y` given
prompt `x`.

Claim: `E(x, y) ≈ -log p(y | x, verifier_consensus)` up to a
calibration constant.

Empirical anchor: post-exp1128 retrain, k=5 ensemble AUROC = 0.94
on FoVer eval. Calibration error is bounded by `|AUROC - 1.0|` (in
the AUROC=1.0 limit, `s` is exactly the posterior probability).

If we treat the verifier ensemble as the agent's generative model
`p(x, z)`, then:

```
F_Carnot[q] = E_q[log q(y|x)] - E_q[log s_ensemble(x, y)]
            ≈ E_q[log q(y|x)] + E_q[E(x, y)]
```

Minimizing `F_Carnot[q]` with respect to `q(y|x)` (the LLM's output
distribution conditioned on the prompt) is exactly: **prefer outputs
the verifier ensemble assigns low energy to**. That is the
verify-repair loop's core operation.

The verify-repair iteration count is therefore the number of
HMC/Langevin steps in active-inference language. The k=5 ensemble's
calibration is the FEP's `p(x, z)` accuracy.

## What's the same vs. different (revised 2026-05-02 07:10Z)

The earlier table treated Phase 4 as a separate paradigm with its
own substrate, generative model, and energy primitive. Per the
score=energy-gradient equivalence, that framing was wrong. The
correct table:

| Component | Carnot today (Phase 1) | Phase 3 prototype | Phase 4 inference modes |
|---|---|---|---|
| **Energy landscape** | k=5 verifier ensemble | + DBAE-EBM | UNCHANGED — same `E(x)` |
| **Energy gradient `∇E`** | Computed on demand | Computed on demand | UNCHANGED — same `∇E` |
| **Sampler** | Langevin / Gibbs / direct rejection | + Langevin on latent | **HMC + diffusion-of-thought** |
| **Inference loop** | Verify → repair | Verify → repair → DBAE refine | + HMC trajectories on `∇E` |
| **Tokens** | LLM tokens | LLM tokens | UNCHANGED |
| **Test-time compute** | Verifier evals + repair iterations | + Langevin steps | + HMC steps + diffusion timesteps |
| **Hardware fit** | KV260 sampler validates `∇E` eval | Z1 thermodynamic Ising | Z1 + photonic (continuous gradient flow) |

**The crucial observation:** every "Phase 4" change is in the right-
most column only. The energy landscape, the verifier ensemble, the
DBAE encoder, the LLM substrate — all unchanged from Phase 3. We
add HMC and diffusion-of-thought as new inference modes alongside
the existing Langevin/Gibbs.

This makes Phase 4 lower-risk than originally framed:
- ✅ No new generative model class
- ✅ No new training pipeline
- ✅ No new verifier suite
- ✅ Reuses Carnot's existing JAX backend (HMC primitives in NumPyro)
- ✅ Falsifiable in 1 milestone instead of 2

## The minimal prototype (revised 2026-05-02 07:10Z)

### Goal

Test whether HMC sampling on Carnot's existing `∇E` (k=5 verifier
ensemble + DBAE-EBM) reproduces Seed IQ's action-efficiency on a
small ARC-AGI-3 subset. Same energy landscape, different sampler.

### Specification

**Energy function — UNCHANGED from Phase 3:**
- `E(x, y) = -log s_ensemble(x, y)` where `s_ensemble` is the k=5
  AND-composed verifier score on candidate output `y` given prompt `x`.
- Post-exp1128: ensemble AUROC = 0.94. Calibration error bounded.
- For ARC-AGI-3: `E` extends to evaluate (state, candidate action)
  pairs by treating the action's predicted outcome as the candidate
  output.

**Gradient `∇_y E(x, y)` — exists natively:**
- Computed via autograd on the verifier ensemble's neural components
  (semantic embedding, ThinkPRM, etc.) and finite-difference for
  symbolic verifiers (Z3, JSON schema). Carnot's existing JAX
  backend supports this.

**HMC sampler — NEW:**
- Numpyro `HMC` primitive on `y` with `∇E` as the negative score.
- Step size, momentum, leapfrog steps tuned per Numpyro defaults.
- Chain length: K HMC steps per action selection. Sweep K ∈ {25, 100, 400}.

**Action selection (ARC-AGI-3 application):**
- For each candidate action: roll out predicted next-state `s'` via
  the environment simulator.
- Compute `E_carnot(x, s')` using the verifier ensemble.
- Use HMC to sample from the posterior `p(action | x) ∝ exp(-E)`.
- Take MAP action (lowest energy in the chain).

**Diffusion-of-Thought mode (separate inference variant):**
- Initialize `y` from random noise.
- Iteratively refine via `y_{t-1} = y_t - η ∇_y E(x, y_t) + noise(σ_t)`
  for T timesteps (forward noise schedule reversed).
- Sweep T ∈ {1, 5, 25, 125} for compute/accuracy curve.

### Acceptance criteria

For 5-10 ARC-AGI-3 puzzles:

| Outcome | Verdict | Implication |
|---|---|---|
| **>50% solve rate** | Hypothesis SUPPORTED | Carnot verifier IS a free-energy approximation; Phase 4 expanded to .92+ |
| **Solve rate matches Phase 3 prototype** | Hypothesis NEUTRAL | Both paradigms work; Carnot positions as synthesis |
| **Solve rate << Seed IQ's 1.00 with similar action count** | Hypothesis WEAK | Verifier-as-FEP-likelihood needs structural changes (e.g., specific verifier per action class) |
| **<10% solve rate** | Hypothesis REJECTED | Active inference + Carnot verifier is not the right combination; Phase 4 reverts to architecture search |

### Falsifiability

The hypothesis is falsifiable by the prototype's solve rate. The
20-iteration HMC budget per action selection bounds the search;
if Carnot's verifier ensemble is the wrong likelihood, the agent
will fail to identify environmental priors and solve rate will be
near zero. Random-baseline comparison at solve rate ≈ chance.

## Decision tree (post-prototype)

```
exp11XX-active-inference-minimal-prototype lands

├── solve rate > 50% on 5-10 puzzles
│   └── Phase 4 EXPANDED to .92+
│       ├── exp11XX-active-inference-full-corpus (50 puzzles)
│       ├── exp11XX-active-inference-hardware-fit (Z1 / photonic)
│       └── v4 paper section: "Carnot is doing active inference"
│
├── solve rate 20-50%
│   └── Phase 4 CONTINUES as exploratory
│       ├── investigate verifier-likelihood mismatch
│       ├── try alternative free-energy approximations
│       └── compare to Phase-3 prototype on same puzzles
│
└── solve rate < 20%
    └── Phase 4 PAUSED
        ├── document hypothesis-rejected outcome
        ├── v4 paper section: "EBM-on-LLM and AI-on-field
        │   are genuinely different paradigms"
        └── re-evaluate at .94+ if Themesis publishes algorithm details
```

## Hardware implications (preserved)

The energy-minimization primitive is invariant across Phase 3 and
Phase 4. Hardware investments stay productive:

- **Extropic Z1**: still the hardware target. Z1's thermodynamic-
  computing pitch maps to Ising-on-EBM (Phase 3) AND to HMC-on-
  continuous-manifold (Phase 4) via the same energy-gradient
  primitive.

- **KV260 sequential sampler** (exp1109's KL=0.025): still
  validates "energy evaluable in dedicated hardware". The sampler
  generalizes from Ising to HMC.

- **Photonic future track**: gets *stronger* under Phase 4. Photonic
  systems natively compute continuous wave dynamics — exactly what
  topological field cognition is. Phase 4 makes the photonic
  argument more direct.

## Cross-references

- `memory/feedback_active_inference_phase4_committed.md` (this directive)
- `memory/project_dbae_ebm_phase3.md` (parallel Phase-3 track)
- `docs/arxiv-paper/main.tex` Section 7 (v3 paper acknowledgment of Themesis)
- `ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES (3 Phase-4 tasks)
- arXiv 2603.24621v1 (ARC-AGI-3 benchmark)
- Karl Friston, "The Free Energy Principle" (foundational)
- arXiv 2024 Renormalising Generative Models (RGMs — Seed IQ underlying tech)
- Themesis, Inc. public ARC-AGI-3 demonstration (2026-04)

## What needs Deep Think (candidate prompt for tomorrow)

The hypothesis "Carnot k=N verifier ≈ free-energy approximation" is
plausible by analogy but not yet proven. A Deep Think prompt could:

1. Walk through the formal mapping: AUROC=0.94 → calibration of the
   posterior approximation
2. Identify which active-inference failure modes Carnot's verifier
   ensemble would or wouldn't catch
3. Specify the prototype's diagnostic instrumentation (per-step
   free-energy estimate, action-policy entropy, prior inference
   confidence)
4. Compare to existing FEP implementations on small benchmarks
   (PyMC + Friston's published toy examples)

Filing as candidate Deep Think Q6 for tomorrow's planning session.

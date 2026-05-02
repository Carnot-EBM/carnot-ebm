# Active Inference Phase-4 Architectural Bridge

**Status:** Committed research direction (2026-05-02 user directive).
Runs as Phase 4 alongside Phase 3 (DBAE-EBM-on-LLM), not as a
replacement. Decision-changing within 1-2 milestones.

**Date drafted:** 2026-05-02
**Source:** Themesis Seed IQ public demonstration on ARC-AGI-3
(0.95 demo + 1.00 reported leaderboard); EBT/ARC-AGI-3 document
positioning Carnot-EBM as a named exemplar of the post-autoregressive
paradigm shift.

---

## The hypothesis

> **Carnot's k=N AND-composed verifier ensemble can serve as a
> calibrated approximation to the variational free-energy lower
> bound under Karl Friston's Free Energy Principle.**

If this hypothesis holds, Carnot's verify-repair loop is
mathematically equivalent to active inference, just framed in
machine-learning terminology rather than computational neuroscience
terminology. Phase 4 tests this directly.

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

## What's the same vs. different vs. Phase 3

| Component | Phase 3 (DBAE-EBM) | Phase 4 (active inference) |
|---|---|---|
| **Energy primitive** | EBM weights `θ` produce `E(x, y)` | Generative model `p(x, z)` produces `F[q]` |
| **Substrate** | LLM base + DBAE encoder + Latent EBM | Continuous belief field on small generative model |
| **Sampler** | Langevin / Gibbs on Ising | HMC on continuous manifold |
| **Inference loop** | Verify → repair → verify | Perception update → action update → re-observe |
| **Verifier role** | Loss / training signal for EBM | Likelihood for `p(x, z)` |
| **Tokens** | LLM tokens | Zero (per Seed IQ design) |
| **Test-time compute** | Refinement steps + verifier evals | HMC chain length |
| **Hardware fit** | Z1 (Ising), KV260 (sampler) | Photonic (continuous wave) |

## The minimal prototype

### Goal

Test whether a tiny active-inference agent (no transformer) using
Carnot's k=N verifier ensemble as the FEP likelihood can match
or approach Seed IQ's published numbers on a small ARC-AGI-3 subset.

### Specification

**Generative model `p(x, z)`:**
- `z`: latent state representing inferred environmental priors
  (invariances, constraints, transition rules, affordances)
- `x`: observation = current grid state + action history
- Parameterization: small Bayesian network or learned latent space
  (e.g., 16-dim continuous vector with structured priors)

**Variational posterior `q(z)`:**
- Gaussian with diagonal covariance (mean-field assumption)
- Initialized from prior; updated via HMC

**Likelihood from Carnot:**
- `p(x | z) ∝ exp(-E_carnot(x, action(z)))` where `E_carnot` is the
  k=N AND-composed energy on the candidate action's predicted
  outcome
- This is the architectural bridge — Carnot's existing infrastructure
  (k=5 ensemble at AUROC=0.94 post-exp1128) provides the likelihood

**Action selection:**
- Sample `K` candidate actions
- Compute `E_carnot(x, y_i)` for each predicted outcome `y_i`
- Take action with lowest expected free energy

**Sampler:**
- PyMC or JAX-NumPyro for HMC on `q(z)`
- ~100-1000 HMC steps per action selection
- No GPU required for prototype scale

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

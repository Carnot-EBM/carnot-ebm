# Phase 3 — Kona Parity Design

**Capability:** phase3-kona
**Version:** 0.1.0 (exploratory design doc, not an implementation plan)

This document is the architectural design behind `spec.md`. It explains **why**
we chose the Recurrent-Depth Transformer (RDT) pattern as the parity architecture,
how the four Kona-parity properties map onto concrete implementation primitives,
and where the honest risks are.

## Why RDT (Recurrent-Depth Transformer)

The most direct architectural route to Kona-style reasoning is one that has
appeared independently in several recent publications: a transformer that
applies a small set of layers iteratively in a recurrent block, rather than
stacking many unique layers.

Three references inform the design (all filed in `research-references.md`):

1. **Looped transformers** (Geiping et al., 2024) — showed that iterative
   application of a small number of layers matches the compute of much deeper
   stacks at lower parameter count.
2. **PonderNet / Adaptive Computation Time** (Banino et al., 2020) — showed
   that a learned halting head allows variable-depth iteration per input,
   improving both compute efficiency and worst-case reliability.
3. **OpenMythos** (2026) — speculative reconstruction of Anthropic's Claude
   Mythos architecture, proposing a three-stage (prelude / recurrent /
   coda) RDT with an LTI stability constraint on the injection matrix.

The LTI stability constraint is the key detail. Without it, iterative
refinement has no convergence guarantee — you can train a model whose training-
set energy trajectory looks nice but whose test-time behaviour diverges on
distribution shift. Constraining the injection matrix's spectral radius to
strictly below 1 gives you a proof (inherited from linear dynamical systems
theory) that the fixed-point iteration converges. That proof is the
difference between a Phase 3 model we can deploy and a curiosity.

**Why not pure diffusion-style generation?** Diffusion models match some
Kona properties (iterative refinement, non-autoregressive generation) but
fail others. They operate on a forward-noising schedule that doesn't match
Carnot's energy-minimisation primitive. RDT is the closer architectural
relative, and its refinement step is literally a gradient step on an
implicit energy function — which is the same primitive Carnot has been
building from since Phase 1.

## The three-stage architecture

```
                 +-----------+    +---------------+    +------+
  Input tokens → |  Prelude  | →  |  Recurrent    | →  | Coda | → Output tokens
                 +-----------+    |  (T steps)    |    +------+
                                  +---------------+
                                     ↑
                                     | re-injected input 'e'
                                     | (LTI-constrained)
```

- **Prelude** (run once). Token embedding and early processing. Outputs an
  initial continuous latent state `h_0` and a reinjection vector `e`.
- **Recurrent block** (run T times, where T is learned or convergence-determined).
  Each step computes `h_{t+1} = A · h_t + B · e + Transformer(h_t, e)`.
  The matrix `A` is LTI-constrained (spectral radius < 1). This is where
  reasoning happens. The same block's parameters are used on every iteration.
- **Coda** (run once at the end). Takes the converged `h_T` and decodes it
  back to tokens. This is the only place tokens are sampled.

The halting decision — when to stop iterating — can come from one of two
places: a **learned halting head** (a small MLP trained on the energy
trajectory) or an **analytic energy convergence check**
(`|E(h_{t+1}) - E(h_t)| < tolerance`). The spec requires both to be
available; in practice the learned head makes the model faster on easy
problems while the analytic check guarantees correctness on hard ones.

## Mapping parity properties onto primitives

| Property | Primitive | Lives in |
|---|---|---|
| Continuous latent reasoning | The recurrent block operates on `jnp.ndarray`, never on token IDs, between prelude and coda. | `python/carnot/phase3/rdt/recurrent_block.py` |
| Non-autoregressive generation | The whole answer is present as a single latent state during refinement; token sampling happens once at the coda, not at each step. | `python/carnot/phase3/rdt/coda.py` |
| Self-correction inside the forward pass | Training loss includes a Phase 1 verify-repair energy term. The model learns to reduce that energy during refinement. | Training harness at `scripts/experiment_*phase3*.py` (not yet written) |
| Hardware portability | The refinement step is a `SamplerBackend` call, not a hard-coded CPU/GPU path. | Existing `python/carnot/samplers/` backend protocol |

## How Phase 1 and Phase 2 feed Phase 3

Phase 3 is not a replacement for Phase 1 and Phase 2 — it consumes their
outputs. Concretely:

- **From Phase 1 (verify-repair):** the training signal for self-correction.
  Phase 1 gives us an energy function we trust. Stage 3 training uses that
  energy as a loss term, teaching the model to produce answers that Phase 1
  would already approve without needing the repair wrapper. If Phase 1's
  energy is broken, Phase 3 will faithfully learn the broken thing. This
  is why the spec lists Phase 1 maturity as a hard dependency.
- **From Phase 2 (hardware):** the deployment target for the refinement
  step. A trained Phase 3 model whose refinement step executes on an FPGA
  Ising sampler or a TSU delivers the original Kona promise — continuous
  latent reasoning that runs on specialised hardware, not on expensive
  CPUs/GPUs. A Phase 3 model that only runs on GPUs is correct but
  off-mission.

## Honest risks

This is where the exploratory nature of the capability matters most.

- **Training cost.** Even at sub-100M-parameter scale, training a
  self-correcting model with a Phase 1 verify-repair loss term is not
  cheap. Stage 3 probably does not fit on the current dual-RTX-3090 setup.
  We do not have a cloud-GPU budget line item today. This is the most
  likely cause of Stage 3 stalling before it starts.
- **Phase 1 reliability as training signal.** The audit discipline that
  caught the "+64 pp" and "0.96 AUROC" retractions was introduced
  *because* Phase 1 numbers were fooling us. If we train Phase 3 against
  an energy that has a blind spot, the model will develop that blind
  spot. Concrete mitigation: only use Phase 1 components that have
  survived cross-dataset audit (the 99.3% wrong-code detector, the typed-
  constraint verifier). Do not use JEPA until it has stable out-of-
  distribution AUC.
- **LTI constraint + expressiveness tradeoff.** Keeping the spectral
  radius strictly below 1 sacrifices some expressive power. If real
  reasoning tasks need dynamics that live near the unit circle (a
  plausible hypothesis for hard problems), the constraint itself becomes
  the limiting factor. Mitigation: the constraint can be relaxed to
  "spectral radius < 1 in expectation" with a penalty term, at the cost
  of losing the clean convergence proof.
- **Halting-head training instability.** Learned halting is known to be
  brittle (PonderNet had to introduce a geometric prior to stabilise it).
  Mitigation: the spec requires the analytic energy-convergence check to
  always be available as fallback, even after the halting head is
  trained.
- **No benchmark exists.** "Kona parity" as defined by the four properties
  is not something the public ML benchmark suite measures directly. Stage
  2 onward will need a benchmark — most likely a synthetic continuous-
  latent reasoning task where the gap between autoregressive and
  refinement-based generation is mechanistically visible. The benchmark
  itself is a deliverable.

## Open design questions

These go to `openspec/change-proposals/` when we start serious work:

- **Injection dimensionality.** Is the reinjected `e` a single vector, a
  set of key-value tensors, or something richer? OpenMythos is vague;
  original RDT papers use a single vector, but there's no theoretical
  reason it couldn't be larger.
- **Prelude / coda depth.** Both are run once, so deeper is cheaper than
  in the recurrent block. Is there a point where making them shallower
  and the recurrent block wider wins?
- **Continuous vs discrete dynamics.** The current formulation is a
  continuous vector equation. Carnot's Ising lineage suggests a
  *discrete-spin* recurrent block (spins iterate toward ground state)
  might map more directly onto FPGA hardware. This is a fork in the road
  worth exploring explicitly before Stage 2.
- **Halting head architecture.** Small MLP on the energy trajectory, or
  something that attends to the latent state itself? Cheapest first.

## Why this is worth writing now, before starting work

The specifications above are intentionally exploratory — `Status: Draft`,
`exploratory design doc, not an implementation plan`. Phase 3 is 12-18
months away from being resource-supportable in the best case. Writing
the spec now does two things:

- **Concretises what "parity" means** so that when someone publishes a
  "Kona-style open-source model", we can evaluate it against these four
  properties rather than argue about scale.
- **Defines the minimum-viable-demonstration task** for Stage 1, which
  is cheap and fits the current hardware. A toy-scale RDT with
  LTI-constrained fixed-point convergence is a 1-week experiment. That
  gets us a primitive whose correctness we can verify, and an early
  concrete argument about whether the LTI constraint is too restrictive
  for reasoning tasks.

Doing Stage 1 soon while Phase 1 hardens and Phase 2 scales is the
honest pragmatic path. The work product is a small Python module and
a failing / passing test artifact — not a foundation model.

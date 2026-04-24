# Phase 3 — Kona Parity Capability Specification

**Capability:** phase3-kona
**Version:** 0.1.0 (exploratory)
**Status:** Draft — primitives only, not yet a shipped capability
**Traces to:** PRD Phase 3 vision (see `_bmad/prd.md`), CLAUDE.md "functional parity with Kona"

## Overview

This capability specifies what it means for Carnot to reach **functional parity with
Logical Intelligence's Kona** — an open-source foundation model whose operating
principles align with Carnot's energy-based lineage. Parity here is defined by four
observable properties, not by parameter count or training compute. An implementation
reaches parity when it can demonstrate all four on a benchmark task, even at small
scale.

The four properties:

1. **Continuous latent reasoning.** The model operates in a continuous latent space
   during its reasoning phase, not in discrete token space. Token decoding happens
   only at the final output step.
2. **Non-autoregressive generation.** Output sequences are produced by iterative
   refinement of the full answer in parallel, not by sampling one token at a time
   from left to right. The inference-time primitive is *energy minimisation over the
   full answer*, not *sample next token given previous tokens*.
3. **Self-correction inside the forward pass.** The verify-and-repair loop that
   Carnot currently implements as an external wrapper (Phase 1
   `VerifyRepairPipeline`) is internalised — the model emits an answer only after
   its own energy has converged below a threshold, or after a bounded number of
   refinement steps.
4. **Hardware portability.** The refinement step maps onto Carnot's existing
   sampler-backend abstraction (`CpuBackend`, `FpgaBackend`, `TsuBackend`,
   `DWaveBackend`). Parity does not require any specific hardware, only that the
   architecture does not pin the model to general-purpose CPUs/GPUs.

This spec is deliberately **capability-level, not implementation-level**. Specific
architectural choices (transformer depth, continuous-latent dimension, halting
mechanism) go in `design.md`. The spec below is the set of observable properties a
concrete implementation has to satisfy.

## Phase 3 dependencies

This capability has **hard upstream dependencies** that are not yet resolved:

- **Phase 1 maturity.** The verify-repair loop must be reliable, well-calibrated,
  and stable enough to serve as the training target for self-correction. Phase 1 is
  currently mature (milestone 2026.04.60 closed), but audit-surfaced retractions
  continue to land in the research record. Phase 3 training on an unreliable
  Phase 1 target would teach the model to reproduce Phase 1's blind spots.
- **Phase 2 hardware scale-up.** Phase 3 training requires fast energy evaluation
  in the refinement loop. At the current 32-spin KV260 prototype scale (live as of
  2026-04-22), the hardware can hold a demonstration problem but cannot serve
  production-scale refinement for a model of interesting size. Either an XCZU-scale
  FPGA, a photonic Ising machine, or an Extropic TSU is the unlock.
- **Dataset and compute.** Kona-style training is unlikely to fit on the current
  dual-RTX-3090 setup. This capability assumes access to cluster-scale compute
  (cloud GPU rental or a research grant) for Stage 3 onward.

Absent these dependencies, Stages 1 and 2 are still valuable — they produce the
architectural primitives and a small-scale demonstration. Stage 3+ is blocked on
the upstream unlocks above.

## Stages

Parity is reached through four stages, each with its own acceptance gate. Earlier
stages are prerequisites for later ones; a stage is not considered started until
the previous stage's gate is passed.

- **Stage 1 — Architecture primitives** (no training).
  Implement the Recurrent-Depth Transformer (RDT) scaffolding in
  `python/carnot/phase3/rdt/`, with LTI-constrained injection, loop-index
  positional embeddings, and an adaptive halting head. Unit tests validate the
  fixed-point convergence property on a synthetic energy landscape.
- **Stage 2 — Tiny-scale end-to-end demonstration** (training on a toy task).
  Train a sub-100M-parameter RDT on a synthetic continuous-latent reasoning task
  (e.g., arithmetic in latent space, or small-graph constraint satisfaction).
  Demonstrate that the model reaches an answer via iterative refinement rather
  than autoregression, and that the number of refinement steps adapts to problem
  difficulty.
- **Stage 3 — Internalised verify-repair** (training on real data).
  Extend Stage 2 with a training signal derived from Phase 1's verify-repair
  pipeline. The refinement step should reduce both the prediction loss and the
  Phase 1 energy score simultaneously. Demonstrate that the trained model's
  single forward pass is competitive with an autoregressive baseline + Phase 1
  verify-repair wrapper at the task.
- **Stage 4 — Hardware-accelerated refinement**.
  Bind the refinement step to one of the Phase 2 sampler backends. Demonstrate
  that the same model checkpoint runs on CPU, GPU, and at least one accelerator
  (KV260 FPGA initially, TSU / photonic when available) with the same observable
  behaviour, only different wall-clock per refinement step.

## Requirements

### REQ-KONA-001: Continuous Latent Refinement

A Phase 3 model MUST maintain its reasoning state as a continuous tensor
(`jnp.ndarray` of float32 or float16) throughout the refinement loop. Token
embeddings enter the continuous state at the prelude stage; token decoding occurs
only at the coda stage. The refinement block MUST NOT sample tokens.

**Rationale:** this is the distinguishing property of Kona-style reasoning.
Discrete-token iteration is autoregressive generation and does not count as Phase 3
parity even if it is iterated.

**Acceptance criteria:**

- `RDTModel.refine_step(state)` has signature `state -> state` where both sides are
  `jnp.ndarray`, not `TokenSequence`.
- A unit test asserts that no `jax.random.categorical` or equivalent token-sampling
  call occurs inside `refine_step`.

### REQ-KONA-002: Bounded-Depth Iterative Refinement

The refinement block MUST be applied a bounded number of times per input (between
`min_steps` and `max_steps`, both configurable), with a halting head that can stop
early when energy convergence is detected.

**Rationale:** unbounded iteration is not a production-viable primitive. The
bound gives the runtime a worst-case guarantee, and the halting head gives the
model agency to stop early on easy inputs.

**Acceptance criteria:**

- `RDTModel.generate(input, min_steps=1, max_steps=64)` returns in at most
  `max_steps` iterations for any input.
- The halting head is trained to predict the energy gradient magnitude; a unit
  test asserts that on a problem with an analytic fixed point, the halting head
  fires within 20% of the true fixed-point step count.

### REQ-KONA-003: LTI Stability Constraint

The injection matrix used to reinject the encoded input at each refinement step
MUST be LTI-constrained — its spectral radius MUST stay strictly below 1.0
during training. This is enforced by spectral-norm regularisation or an equivalent
constraint.

**Rationale:** without this constraint, iterative refinement can diverge. The
spectral-radius bound is the standard stability proof for linear dynamical
systems and is cheap to enforce with a regularisation term.

**Acceptance criteria:**

- `LTIInjectionLayer.effective_spectral_radius()` returns a float < 1.0 at every
  training step.
- A unit test asserts that on a synthetic diverging initialisation, the
  constraint training loop brings the spectral radius below 1.0 within 100 steps.

### REQ-KONA-004: Energy-Convergence Halting Criterion

The halting decision MUST be triggered either by the adaptive halting head (learned
signal) or by a direct energy-convergence check (analytic signal): the refinement
loop stops when `|E(state_t) - E(state_{t-1})| < tolerance` for a configurable
tolerance.

**Rationale:** a learned halting head is a useful training signal but not a
reliability guarantee. The analytic energy-convergence check is the principled
stopping condition and MUST be available as a fallback.

**Acceptance criteria:**

- `RDTModel.generate(..., use_learned_halting=False)` stops on the energy
  criterion alone.
- Both modes agree on the stopping step within 10% on a validation set.

### REQ-KONA-005: Internalised Verify-Repair Signal

A Stage 3+ Phase 3 model MUST include a training loss term derived from Carnot's
Phase 1 verify-repair pipeline. The refinement step is rewarded for reducing
Phase 1's violation energy alongside the primary prediction loss.

**Rationale:** this is what moves the verify-repair loop from "external wrapper"
to "internal behaviour". Without it, Stage 3 produces a pretty RDT that still
needs Phase 1 around it — which is not parity.

**Acceptance criteria:**

- The training loss includes a `phase1_energy_weight * phase1_violation_energy`
  term with `phase1_energy_weight > 0`.
- An ablation unit test confirms that training with this term produces a model
  whose single forward pass has strictly lower Phase 1 energy than training
  without it, on a held-out set.

### REQ-KONA-006: Sampler-Backend Compatibility

The refinement step MUST be implementable as a call to any of the
`SamplerBackend` implementations in `python/carnot/samplers/`. Swapping backends
MUST NOT change the model's observable output distribution (within numerical
tolerance), only the wall-clock per step.

**Rationale:** hardware portability is one of the four parity properties. It also
makes the capability honest — if Phase 3 is irreversibly wedded to CPU/GPU, it
has not actually used Carnot's hardware lineage.

**Acceptance criteria:**

- A test runs the same checkpoint through `CpuBackend` and at least one other
  backend on a small validation set and asserts output KL-divergence < 0.01.
- FpgaBackend support is exercised when the KV260 is available.

### REQ-KONA-007: Honest-Verdict Emission

Phase 3 experiments MUST populate the `honest_verdict` schema field with one of:
`stage1_primitives_only`, `stage2_toy_converged`, `stage2_toy_diverged`,
`stage3_verify_repair_internalised`, `stage3_verify_repair_regressed`,
`stage4_backend_swap_verified`, `stage4_backend_swap_failed`, or
`blocked_*` for dependency failures.

**Rationale:** Phase 3 is where optimistic claims are most tempting. The
`honest_verdict` discipline that caught the Phase 1 retractions (the "+64 pp VR",
the cross-dataset 0.96 AUROC, the 1.0 JEPA OOD AUC) must extend here.

**Acceptance criteria:**

- Any experiment script under `scripts/experiment_*phase3*.py` or
  `scripts/experiment_*kona*.py` asserts at exit that its artifact contains a
  non-empty `honest_verdict` matching the enum above.

## Scenarios

### SCENARIO-KONA-001: Stage 1 Primitive — RDT Fixed-Point Convergence

**Given** a trained `RDTModel` on a synthetic energy landscape with a known global
minimum
**When** `RDTModel.refine(state_0, max_steps=100)` is called from an arbitrary
initialisation
**Then** the returned state is within `tolerance` of the global minimum for all
initialisations in a validation cohort

**Spec traces:** REQ-KONA-001, REQ-KONA-002

### SCENARIO-KONA-002: LTI Constraint Holds During Training

**Given** an `RDTModel` with an `LTIInjectionLayer`
**When** 1,000 training steps are executed on a synthetic task
**Then** `effective_spectral_radius()` is strictly less than 1.0 at every
checkpoint

**Spec traces:** REQ-KONA-003

### SCENARIO-KONA-003: Learned and Analytic Halting Agree

**Given** a trained Stage 2 `RDTModel` with both halting modes available
**When** `generate(input, use_learned_halting=True)` and
`generate(input, use_learned_halting=False)` are called on 100 validation inputs
**Then** the two modes' stopping-step counts agree within 10% mean absolute error

**Spec traces:** REQ-KONA-004

### SCENARIO-KONA-004: Verify-Repair Loss Improves Phase 1 Score

**Given** two Stage 3 checkpoints trained with and without the
`phase1_energy_weight` term, all other hyperparameters equal
**When** both are evaluated on a held-out verify-repair benchmark
**Then** the model trained with the term has a lower mean Phase 1 violation
energy on single-forward-pass outputs

**Spec traces:** REQ-KONA-005

### SCENARIO-KONA-005: Backend Swap Preserves Output Distribution

**Given** a Stage 4 `RDTModel` checkpoint
**When** the same input batch is processed through `CpuBackend` and a second
backend (GPU or FpgaBackend when available)
**Then** the per-sample output distributions have KL divergence < 0.01

**Spec traces:** REQ-KONA-006

### SCENARIO-KONA-006: Non-Parity Attempt Emits Honest Verdict

**Given** a Stage 2 experiment where `refine_step` secretly calls
`jax.random.categorical`
**When** the acceptance test for REQ-KONA-001 runs
**Then** the test fails, the experiment artifact records
`honest_verdict='stage2_toy_diverged'`, and the run is not considered passing

**Spec traces:** REQ-KONA-001, REQ-KONA-007

## Out of scope

The following are deliberately **not** required by this capability:

- **Parameter-count parity with Kona's published size.** Parity is
  architecture-level, not scale-level. A 100M-parameter Phase 3 model that
  exhibits all four properties counts; a 35B-parameter autoregressive transformer
  does not.
- **Public-benchmark leadership.** Demonstrating Phase 3 parity on a toy task is
  sufficient for this capability; surpassing published benchmarks is a Phase 4
  ambition not captured here.
- **Tokenizer innovation.** Kona-style reasoning happens in continuous latent
  space after tokenization; the tokenizer itself is orthogonal.
- **Training-data innovation.** Whatever dataset trains Phase 1's verify-repair
  well enough to serve as the Stage 3 target is sufficient.

## Implementation status

- **Stage 1 primitives:** partial.
  `python/carnot/phase3/continuous_ebm.py` exists (Exp 435a) with
  `ContinuousEBMMinimiser`, Langevin and energy-matching samplers (Exp 446).
  The RDT scaffolding is **not yet written**. The LTI constraint layer is **not
  yet written**.
- **Stage 2 demo:** not started.
- **Stage 3 internalisation:** not started (Phase 1 maturity dependency).
- **Stage 4 hardware binding:** gated on Phase 2 scale-up.

First concrete next experiment:
`experiment_XXX_rdt_primitive_convergence.py` — implement the RDT scaffold and
the LTI-constrained injection, verify SCENARIO-KONA-001 (fixed-point convergence
on synthetic landscape) and SCENARIO-KONA-002 (LTI constraint holds). Expected
to be a 1-week effort; would not require GPU beyond what's already available.

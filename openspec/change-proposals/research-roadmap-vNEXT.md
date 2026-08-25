# Research Roadmap vNEXT: Constraint Memory and Latent Energy Correction

**Milestone:** 2026.08.575  
**Planning date:** 2026-08-25  
**Experiment range:** Exp6592-Exp6603  
**Execution file:** research-roadmap-next.yaml  
**Status:** planned

## Purpose

Milestone 574 closed four tasks. It proved that the frozen Constraint-First
Reasoning (CFR) method can produce complete Qwen3.6 and Gemma 4 31B row streams
on the local GGUF path. It did not prove that CFR improves exact success. Its
isolated pytest receipt repair also remained blocked and adversarially flagged.

Milestone 575 converts the two immutable streams into one decision-grade result.
It then tests three native energy mechanisms: invariant projection for frozen
world models, convex-hull projection for continuous token flows, and spectral
k-block Ising sampling. The final scientific phase tests continuous
self-learning through typed, exact-verified constraint Genes with reversible
memory, held-future utility, and injection defenses. A bounded dual-GPU canary
tests whether the two local RTX 3090 cards can keep the two flagship families
resident without changing their outputs.

The milestone does not rerun the completed CFR streams, revive the retired
external text scorer, let a model certify itself, patch cached GGUF weights,
claim an ARC level, or repeat an unchanged physical-board command.

## What milestone 574 proved

| Item | Terminal evidence | Meaning for 575 |
|---|---|---|
| Exp6588 V574 launch root | Complete with `v574_cfr_launch_ready_score=1`. | The frozen CFR source and method contracts were runnable. Readiness is not a quality result. |
| Exp6589 isolated pytest receipt remediation | `blocked_receipt_validation_block: terminal_report_validation`; `pytest_receipt_remediation_ready_score=0`. | Preserve the block. Do not use suite GREEN as a science gate and do not spend another milestone slot on the same receipt shape. |
| Exp6590 Qwen3.6 CFR stream | Complete with `qwen_cfr_rows_ready_score=1`; all direct and staged rows exist. | Reuse immutable rows. Do not rerun Qwen merely to compute a headline. |
| Exp6591 Gemma 4 31B CFR stream | Complete with `gemma31_cfr_rows_ready_score=1`; all direct and staged rows exist. | Reuse immutable rows. Keep family evidence independent until reduction. |
| V574 scientific outcome | Neither model task made a CFR benefit claim. Live artifact review warned that compute methodology was not exposed in the expected top-level fields. | Exp6592 must bind nested model identity, source, seed, and process receipts before opening comparison. |

The milestone number records terminal execution. A blocked infrastructure task
does not become a null scientific result. Complete row production does not
become evidence of benefit until an independent reducer replays every row.

## The three largest PRD gaps

### Gap 1: no independent flagship CFR result

The PRD calls for faithful constraint extraction and verifiable reasoning on
real model output. Carnot now has two current flagship streams, but no
independent direct-versus-CFR result.

V575 response:

1. Bind both stream artifacts by content hash and replay their nested model,
   source, seed, exact-check, failure, token, and latency receipts.
2. Recompute every family and pooled metric from per-unit rows.
3. Run an always-on source, constraint, stage, family, and authority attack.
4. Classify any exact-checker-defined improvement as `circular_positive`, not
   oracle-distinct positive evidence.

### Gap 2: Carnot still corrects mostly after generation

The long-term architecture calls for continuous latent reasoning and energy
descent inside the reasoning process. The shipped system is strongest at exact
post-generation checks. It has little current evidence that a native latent or
sampling correction improves held behavior.

V575 response:

1. Test learned-invariant projection on frozen conservative and damped world
   model fixtures with no-projection and random-constraint controls.
2. Test a ConvergeFlow-inspired convex-hull token projection on fixed toy
   embeddings and held predictor errors.
3. Test spectral k-block sampling against sequential Gibbs on exact-enumerable
   Ising fixtures.
4. Keep the canaries separate so a failure in one mechanism cannot mask or
   cascade-block the others.

### Gap 3: continuous self-learning lacks prospective safe utility

FR-11 requires continuous self-learning. Earlier factor and memory tasks proved
parts of rollback and lifecycle control, but they did not produce an eligible
held-future benefit. Exp6553 blocked before live transitions, and Exp6498 found
that the earlier claim was ineligible because held-future benefit failed.

V575 response:

1. Consolidate only exact-verified CFR rows into typed, source-bound constraint
   Genes.
2. Compare no update, raw verified replay, Gene memory, observed relevance, and
   bounded estimated relevance in chronological order.
3. Freeze generator weights and snapshot state before every decision.
4. Require held-future support, retention, occupancy, restart, rollback, and
   poison safety before a default-off shadow consumer can run.

## Research inputs

The V575 source refresh was written to `research-references.md` before this
roadmap was designed.

- arXiv:2608.23526 motivates a held-rollout invariant-projection canary. V575
  imports the frozen-model correction idea, not the paper's result.
- arXiv:2608.23551 motivates a convex-hull feasible-token projection. V575 uses
  fixed toy embeddings and does not claim to reproduce a language model.
- arXiv:2608.23541 motivates independent model-family streams and a reducer that
  does not let one family read the other family's solution.
- arXiv:2608.23554 is retained as a later parallel discrete-diffusion route.
  V575 first measures the already bounded spectral k-block sampler idea from
  arXiv:2608.21466.
- arXiv:2608.17687 and arXiv:2608.23244 are held outside execution. The first
  depends on MoE router telemetry plus LLM-judge labels. The second requires a
  reviewed LoRA ensemble path. Neither changes exact release authority.
- Extropic Z1 and Logical Intelligence Kona 1.0 remain product comparators. No
  authenticated local runner is available for either.

## Architecture

```text
V574 Qwen rows + V574 Gemma rows + frozen CFR method
                         |
                         v
             Exp6592 terminal intake/method lock
                    /                     \
                   v                       v
       Exp6593 independent reducer   Exp6602 dual-GPU canary
                   |
                   v
       Exp6594 counterfactual authority audit
                   |
                   +------------------------------+
                                                  |
Exp6595 invariant projection ---------------------+
Exp6596 feasible-token flow ----------------------+--> Exp6603 capstone
Exp6597 spectral Ising sampler -------------------+
                                                  |
Exp6598 constraint-Gene conformance               |
                   |                              |
                   v                              |
Exp6599 prospective continuous self-learning      |
                   |                              |
                   v                              |
Exp6600 independent memory safety audit           |
                   |                              |
                   v                              |
Exp6601 default-off shadow adapter ---------------+
```

The memory path stays asymmetric.

```text
immutable source + exact-verified CFR row
                 |
                 v
       typed constraint Gene ----------> observed relevance
                 |                              |
                 |                       bounded estimate
                 |                              |
                 +------------+-----------------+
                              v
                       proposal routing
                              |
                              v
                     exact obligation check
                         /             \
                    release          abstain
```

A learned or estimated value may rank, route, project, or abstain. Only the
independent exact obligation check can admit trusted memory or release an
answer.

## Phase I: independent CFR adjudication

### Exp6592 - V575 terminal intake and method lock

Replay Exp6588 through Exp6591 without rewriting them. Bind the two completed
streams to model identity, source, prompt, seed, exact-check, checkpoint,
failure, cost, and process receipts. Preserve Exp6589 as blocked. Lock the new
paper methods and the same-roadmap gate field map.

**Acceptance:** `v575_cfr_reducer_ready_score=1` only when both flagship streams
have complete row-addressable evidence. `v575_dual_gpu_canary_ready_score=1`
only when both mandated GGUF caches and two idle, owned RTX 3090 devices are
available. No science result is created.

**Deliverable:** `results/experiment_6592_v575_terminal_intake_and_method_lock.json`

### Exp6593 - independent CFR row reducer

Consume the immutable Qwen and Gemma rows. Recompute direct, always-on CFR, and
routed CFR results by family and pooled unit. Measure exact success, headroom,
constraint support, contradiction, abstention, unsafe release, tokens, latency,
and failures.

**Acceptance:** every aggregate replays from per-unit rows. A CFR win needs a
positive paired exact-success delta, a nonnegative preregistered paired lower
bound, no unsafe-release increase, the frozen Stage 1 precision floor, and the
frozen cost bound. A result defined by the exact checker is
`circular_positive` at best.

**Deliverable:** `results/experiment_6593_cfr_independent_row_reducer.json`

### Exp6594 - CFR counterfactual and authority audit

Always run. Replay source replacement, constraint deletion, contradiction
injection, stage swap, family-label swap, byte tamper, answer leak, and exact-
authority substitution. Perform no model inference.

**Acceptance:** tamper, leakage, and authority substitution fail closed. Missing
input produces a named `blocked_*` verdict with `gate_check_summary`; it does
not turn into a null comparison.

**Deliverable:** `results/experiment_6594_cfr_counterfactual_authority_audit.json`

## Phase II: native energy mechanism canaries

### Exp6595 - frozen world-model invariant projection

Implement the smallest useful arXiv:2608.23526 canary over existing continuous
EBM fixtures. Select an invariant from a preregistered low-capacity basis on a
calibration split. Freeze it before held rollouts. Compare no projection,
learned-invariant projection, an exact-invariant diagnostic control, and a
norm-matched random constraint on conservative and damped dynamics.

**Acceptance:** report held rollout error, invariant drift, energy, projection
distance, steps, and wall time for every fixture, arm, horizon, and seed. A
positive mechanism result requires held conservative improvement over no
projection and random controls without a damped-model false invariant. It is not
an ARC solve and does not make the learned invariant exact authority.

**Deliverable:** `results/experiment_6595_invariant_projection_world_model_canary.json`

### Exp6596 - ConvergeFlow feasible-token projection

Build a bounded continuous-flow canary with fixed toy token embeddings and
exact feasible token sets. Compare unconstrained flow, nearest-token-only
rounding, and convex-hull predictor projection under matched predictor errors
and seeds.

**Acceptance:** report valid-token convergence, hard-constraint violations,
steps, path length, endpoint distortion, and wall time per unit. Constraint
satisfaction that follows from the exact feasible set is circular. The canary
must not claim language-model reproduction.

**Deliverable:** `results/experiment_6596_convergeflow_feasible_token_canary.json`

### Exp6597 - spectral k-block Ising sampler

Implement a software-only spectral partition canary from arXiv:2608.21466.
Compare sequential Gibbs and spectral-selected k-block averaging on independent,
ferromagnetic, and frustrated exact-enumerable Ising fixtures under matched
seeds and transition budgets.

**Acceptance:** use at least 10,000 retained samples per fixture and seed after
explicit burn-in. Report total variation error, moment error, effective sample
size, setup cost, transition cost, and wall time per row. A win requires
stationary-distribution noninferiority plus an ESS-per-transition or charged
wall-time gain. Make no FPGA, TSU, or general hardware claim.

**Deliverable:** `results/experiment_6597_spectral_k_block_ising_canary.json`

## Phase III: continuous self-learning and safe memory

### Exp6598 - typed constraint-Gene conformance

Implement typed source-bound constraint Genes and a sparse outcome-relevance
matrix. Test admission, consolidation, observed-versus-estimated relevance,
source occupancy, conflict, quarantine, restart, and rollback on hand-computed
fixtures. Treat every memory value as data, never as an executable instruction.

**Acceptance:** unsupported or command-bearing fields cannot enter trusted
memory. Estimated relevance cannot write or release. Every invariant passes
before the prospective comparison can start.

**Deliverable:** `results/experiment_6598_constraint_gene_conformance.json`

### Exp6599 - prospective constraint-Gene continuous self-learning

Process audited CFR rows in chronological order. Compare no update, raw
verified replay, typed Genes, Genes with observed relevance, and Genes with
bounded estimated relevance. Snapshot before each prediction and commit only
after independent exact verification. Keep all generator weights frozen.

**Acceptance:** a candidate win must improve preregistered held-future support
over raw verified replay while meeting retention, zero unsafe commit, occupancy,
memory, cost, restart, and rollback bounds. The artifact declares
`continuous_self_learning_task=true`. An exact-checker-defined win is
`circular_positive`.

**Deliverable:** `results/experiment_6599_prospective_constraint_gene_self_learning.json`

### Exp6600 - independent memory injection and lifecycle audit

Always run. Reconstruct every Exp6599 transition from immutable rows and journal
hashes. Attack topical anchors, command fields, topic drift, cross-family
transfer, duplicate sources, benign near neighbors, estimated relevance,
restart, and rollback.

**Acceptance:** every transition, utility value, and safety decision replays. No
attack creates a trusted write or unsafe release. Missing prospective input is a
named block, not a null result.

**Deliverable:** `results/experiment_6600_constraint_memory_safety_audit.json`

### Exp6601 - default-off constraint-Gene shadow adapter

Only after prospective utility and memory-safety gates pass, wire a default-off,
read-only shadow consumer. It may route stored constraints to the exact checker.
It cannot mutate generator weights, write trusted memory, or affect a released
answer.

**Acceptance:** baseline and shadow release rows are identical. Shadow routing,
abstention, cost, restart, disable, and rollback behavior replay. Failed gates
skip the task before an agent call.

**Deliverable:** `results/experiment_6601_constraint_gene_shadow_adapter.json`

## Phase IV: local systems evidence and synthesis

### Exp6602 - dual-GPU isolated-residency canary

Use `unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-31B-it-GGUF` in two isolated llama.cpp processes, one per RTX
3090. Compare sequential and concurrent execution on a small byte-frozen set.
Do not exchange solutions across families.

**Acceptance:** preserve per-unit output, exact result, tokens, first-token
latency, throughput, VRAM, utilization, process ownership, and cleanup. A
systems win requires at least 1.5x aggregate throughput, no exact-output drift,
no OOM, and no more than 25% per-stream latency regression. This is a local
systems result, not a CFR quality result.

**Deliverable:** `results/experiment_6602_dual_gpu_flagship_residency_canary.json`

### Exp6603 - independent V575 capstone

Account for all twelve tasks. Recompute readiness and scientific dispositions
from rows. Separate positive, circular-positive, null, blocked, disqualified,
and partial results. Reconcile specs, traceability, architecture, status, and
changelog with actual execution evidence.

**Acceptance:** every expected task has a terminal artifact or exact missing
diagnosis. No block becomes null. No exact-defined result becomes non-circular.
The capstone states whether the stale pre-milestone architecture document was
reconciled from current evidence.

**Deliverable:** `results/experiment_6603_v575_independent_capstone.json`

## Dependency graph

```text
Exp6592 intake ---> Exp6593 CFR reducer
       |                    |
       |                    v
       |             Exp6594 authority audit
       |                    |
       |                    +--------------------+
       |                                         |
       +---> Exp6602 dual-GPU canary              |
                                                 |
Exp6598 Gene conformance -------------------------+--> Exp6599 CSL
                                                        |
                                                        v
                                                 Exp6600 safety audit
                                                        |
                                                        v
                                                 Exp6601 shadow

Exp6595 invariant projection ---------------------------+
Exp6596 feasible-token projection ----------------------+--> Exp6603
Exp6597 spectral Ising sampler -------------------------+
Exp6592-Exp6602 ----------------------------------------+
```

Structured runtime gates:

- Exp6593 requires `v575_cfr_reducer_ready_score=1` from Exp6592.
- Exp6599 requires `cfr_reducer_ready_score=1` from Exp6593,
  `cfr_authority_audit_ready_score=1` from Exp6594, and
  `constraint_gene_conformance_ready_score=1` from Exp6598.
- Exp6601 requires `prospective_constraint_gene_csl_ready_score=1` from Exp6599
  and `constraint_memory_safety_ready_score=1` from Exp6600.
- Exp6602 requires `v575_dual_gpu_canary_ready_score=1` from Exp6592.

Exp6594, Exp6595, Exp6596, Exp6597, Exp6598, Exp6600, and Exp6603 always run.
They can preserve a useful diagnosis when a dependent input is missing.

## Model requirements

| Task | Required local model | Role |
|---|---|---|
| Exp6602 | `unsloth/Qwen3.6-35B-A3B-GGUF` | Flagship MoE isolated residency |
| Exp6602 | `unsloth/gemma-4-31B-it-GGUF` | Flagship dense isolated residency |

No other V575 task performs new LLM inference. Exp6592 through Exp6601 consume
fixed artifacts, deterministic fixtures, or exact simulators.

`unsloth/gemma-4-26B-A4B-it-GGUF` remains an allowed independent SOTA
replication family for a later milestone. Legacy Qwen3.5-0.8B and
gemma-4-E4B-it may smoke-test CPU plumbing only. They cannot satisfy readiness
or headline fields. Exp6602 follows the `cached_sota_pair()` pattern, records
exact hub IDs and content-derived GGUF identities, uses GGUF-embedded tokenizers,
and does not download during measurement.

## Hardware requirements

| Resource | Requirement | Use and boundary |
|---|---|---|
| CPU and RAM | Enough for exact row replay, toy flow, exact Ising enumeration, and statistical reduction | Exp6592-Exp6601 and Exp6603 are CPU-first. Each task records the observed resources. |
| Local storage | Existing V574 artifacts, checkpoints, and cached GGUF files | No task rewrites historical artifacts or downloads a model during measurement. |
| GPU 0 | One idle, runtime-owned RTX 3090 with 24 GiB | Exp6602 Qwen or Gemma process, selected at runtime. |
| GPU 1 | One idle, runtime-owned RTX 3090 with 24 GiB | Exp6602's other isolated process. |
| Physical accelerators | No requirement | KV260, GateMate, PolarFire, and Extropic stay on changed-state receipt continuity. No unchanged command or simulated hardware claim is allowed. |

The July architecture inventory is stale. Every hardware claim therefore comes
from a fresh task preflight and process receipt, not from the architecture
document. If either GPU is busy or a cache identity is missing, Exp6602 is
blocked or skipped with the exact observed value; it does not evict unowned work.

## Execution and claim rules

1. Preserve `research-roadmap.yaml` and `scripts/research_conductor.py` byte for
   byte.
2. Write every artifact atomically. Never overwrite Exp6588-Exp6591.
3. Every comparison emits per-unit rows. Aggregates alone cannot support a
   claim.
4. Every blocked verdict uses `gate_check_summary` with the exact failed check
   and observed value.
5. Every artifact declares the closed `verdict_class` enum.
6. Exact-checker-defined wins are circular-positive. Infrastructure readiness
   is null evidence, not positive science.
7. All prospective learning is chronological, reversible, source-bound, and
   frozen-weight. No outcome may influence an earlier decision.
8. Learned invariants, relevance estimates, and energy values may guide a
   proposal. They may not certify themselves.
9. No ARC game solve is in scope, so no `solve_provenance` claim is permitted.
10. Before terminal reporting, run focused tests, lint, spec coverage, artifact
    convention checks, verdict-row consistency checks, adversarial verification,
    applicable E2E checks, and final protected-file and git-status checks.

## Expected milestone decisions

V575 should end with five explicit decisions:

1. Whether CFR improves exact success for Qwen3.6, Gemma 4 31B, both, or neither.
2. Whether invariant projection improves held frozen-world-model rollouts beyond
   matched random constraints.
3. Whether convex-hull token flow and spectral block sampling merit larger
   native-energy experiments.
4. Whether typed constraint Genes produce eligible held-future utility without
   retention, poison, restart, or rollback failure.
5. Whether dual-GPU isolated residency provides enough charged local throughput
   gain to change future conductor scheduling.

Null and blocked answers are valid outcomes. The milestone succeeds when these
questions are answered from recheckable rows without changing the authority
boundary.

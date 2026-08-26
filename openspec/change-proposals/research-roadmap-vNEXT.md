# Research Roadmap vNEXT: Executable Constraints and Verified Skill Evolution

**Milestone:** 2026.08.579  
**Status:** Planned  
**Tasks:** 14  
**Execution file:** `research-roadmap-next.yaml`  
**Research basis:** `research-references.md`, V579 planner refresh

## What milestone 2026.08.578 proved

V578 ended terminally, but it did not test its scientific hypotheses. Its design
document named Exp6619 through Exp6632. The active YAML named only Exp6619
through Exp6628. Exp6619 detected that mismatch and set its readiness score to
zero. The dependent execution and science tasks then blocked or skipped.

That result proves one operational point: a roadmap must be one prevalidated
execution unit. It does not prove that the GPU lease, mandated models, delayed
constraints, spectral sampler, invariant memory, or continuous self-learning
hypotheses fail.

V576 still supplies the last scientific evidence:

| Area | Evidence retained | Limit |
|---|---|---|
| Direct flagship baselines | Model files and partial process evidence exist | No complete, owned, replayable candidate rows |
| Delayed two-level constraints | Exact corpus and method contract exist | No eligible flagship treatment run |
| Live ARC invariant projection | The live `E3AgentPolicy` path was reached | Projection changed no action and earned no solve credit |
| Spectral k-block sampling | Software rows suggested transition and wall-time gains | Reference parity, protection, and independent replay did not pass |
| Invariant memory | Lifecycle and rollback contracts passed | Prospective arm had no credited future benefit |

## Three largest gaps to the PRD vision

### Gap 1: no execution-admitted flagship evidence

The PRD needs local, privacy-preserving reasoning on useful models. Carnot has
the mandated GGUF files, but it lacks one proven task-scoped path from lease to
load, resident inference, unload, and atomic evidence. A model result is not
admissible until identity, process, phase, accelerator, and row receipts share
one owner.

### Gap 2: constraints guide proposals, but exact authority is not yet closed

Carnot has exact checkers, constraint-first streams, frozen candidate pools, and
sampler prototypes. It does not yet have a complete flagship comparison in
which direct generation establishes headroom, delayed syntax and semantic
constraints preserve support, the verifier unit is calibrated on clean twins,
and an independent exact audit owns release. The Ising branch also lacks an
independent exact reference for its speed claims.

### Gap 3: self-learning state is stored but does not improve future live action

The PRD requires continuous learning with provenance, rollback, and safety.
Carnot can store and retire invariant memory, yet it has not shown that stored
state survives handoff as a binding constraint, changes a live held-out ARC
action, repairs an independently located component, and improves later events
across task orders. Same-event adaptation and same-model self-grades do not
close this gap.

## Research questions

1. Can a task-scoped dual-RTX execution envelope admit all three mandated GGUF
   families without reconstructed or shared process evidence?
2. Does delayed two-level constraint guidance improve exact validity while
   preserving candidate support on complete flagship rows?
3. Which verification unit improves discrimination rather than merely raising
   rejection?
4. Can a Kac--Ward exact planar reference support an honest local spectral
   sampler replay?
5. Can operationally complete memory change a held-out live ARC action and then
   improve future events under exact update authority?

## Architecture under test

```text
                         exact authority plane
             +-----------------------------------------+
             | corpus labels | twin labels | Kac--Ward |
             | ARC outcomes  | support audit | rollback|
             +-----------+-------------+---------------+
                         |             |
                         v             v
+------------------+   +------------------------+   +------------------+
| local GGUF models|-->| proposal and search    |-->| checked candidate |
| Qwen3.6 35B A3B  |   | direct / delayed       |   | pool and release  |
| Gemma 4 31B      |   | syntax / semantic      |   | decision          |
| Gemma 4 26B A4B  |   +-----------+------------+   +------------------+
+--------+---------+               |
         ^                         v
         |             +--------------------------+
         |             | live E3AgentPolicy       |
         |             | held-out state/action    |
         |             +------------+-------------+
         |                          |
         |                          v
         |             +--------------------------+
         +-------------| verified skill memory    |
                       | invoke / revise / retire  |
                       | restart / rollback        |
                       +--------------------------+

Execution envelope: task lease -> model identity -> phase journal -> rows
Sampler branch: planar instance -> Kac--Ward reference -> CPU/Rust/RTX replay
```

The exact plane owns admission and release. Learned models may propose, score,
route, or abstain. A learned signal never certifies itself.

## Phase A: Admit real local inference

### Exp6633 - Task-scoped GPU lease and phase journal

Implement the reusable lease that V577 and V578 never reached. Bind task, token,
device UUID, PID start, model identity, heartbeat, phase, VRAM, exit, and unload.
Test races, PID reuse, stale owners, tamper, restart, and protected files without
loading an LLM.

- Deliverable: `results/experiment_6633_gpu_lease_phase_journal.json`
- Gate output: `gpu_lease_scheduler_ready_score`
- Prior scopes: Exp6617 and Exp6620
- Hardware: bounded process fixtures only

### Exp6634 - Independent admission of all mandated GGUF families

Run three fresh-process canaries under independent leases. Admit each family
separately. A failure in one family must not erase valid evidence for another.

- Models: `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`
- Deliverable: `results/experiment_6634_mandated_model_admission.json`
- Gate outputs: `qwen36_admitted_score`, `gemma4_31b_admitted_score`, and
  `gemma4_26b_admitted_score`
- Hardware: dual RTX 3090; one family per fresh process

## Phase B: Close the constrained-generation authority loop

### Exp6635 - Matched direct-headroom baselines

Run frozen direct prompts on Qwen3.6-35B-A3B and Gemma-4-26B-A4B. Emit every
candidate row, exact validity, support, model identity, and process receipt.
Open the treatment gate only if complete rows show preregistered error headroom.

- Deliverable: `results/experiment_6635_matched_direct_headroom.json`
- Gate outputs: `direct_rows_complete_score`, `headroom_ready_score`
- Prior scopes: Exp6605, Exp6607, and Exp6622

### Exp6636 - Delayed two-level constrained decoding

Compare direct, syntax-only, and delayed syntax-plus-semantic guidance on one
frozen pool specification and fixed budgets. Keep an unconstrained reasoning
region. Measure exact validity, diversity, best-at-k support, latency, and
constraint activation per unit.

- Deliverable: `results/experiment_6636_delayed_two_level_decoding.json`
- Gate output: `decoding_rows_complete_score`
- Prior scopes: Exp6609 and Exp6624

### Exp6637 - Cold-context verifier unit calibration

Use byte-matched clean/error twins. Vary only the reviewed action unit. Compare
fresh, prior-repair, and length-matched contexts. Report catch, false rejection,
informedness, criterion, and discrimination. Exact labels own the answer.

- Models: Qwen3.6-35B-A3B and Gemma-4-31B
- Deliverable: `results/experiment_6637_verifier_unit_calibration.json`
- Research anchors: arXiv:2608.23941 and arXiv:2608.16003

### Exp6638 - Independent exact release audit

Recompute all Exp6636 validity and support fields from frozen candidate rows.
Compare the learned routing signal with exact outcomes, verify that rejected
valid candidates remain counted, and issue the only release decision for the
decoding branch.

- Deliverable: `results/experiment_6638_exact_decoding_authority_audit.json`
- Gate output: `exact_decoding_release_score`

## Phase C: Establish an exact sampling reference before acceleration

### Exp6639 - Kac--Ward planar-Ising reference

Implement a bounded exact autoregressive sampler for small zero-field planar
Ising instances. Cross-check normalized likelihoods and moments against full
enumeration. Generate independent reference rows for later sampler comparison.

- Deliverable: `results/experiment_6639_kac_ward_planar_reference.json`
- Gate output: `kac_ward_reference_ready_score`
- Research anchor: arXiv:2608.24382
- Prior scope: Exp6612, addressed by a new independent exact technique

### Exp6640 - CPU, Rust, and RTX spectral replay

Replay the k-block method only after the exact reference passes. Compare
stationary error, transition cost, setup, wall time, effective sample size,
autocorrelation, and protection receipts on matched planar rows. Scope the claim
to measured local paths. Do not make FPGA or TSU claims.

- Deliverable: `results/experiment_6640_spectral_local_replay.json`
- Prior scopes: Exp6612 and Exp6627
- Hardware: CPU, Rust implementation, and local RTX 3090 CUDA path

## Phase D: Make verified memory affect future live action

### Exp6641 - Operational-state handoff preservation

Compare direct handoff, ordinary compression, and a typed four-field handoff for
live-policy memory. Measure prerequisite, authority, fallback, consequence,
deactivation, and forbidden action per episode. Semantic mention alone does not
count as preservation.

- Model: Qwen3.6-35B-A3B
- Deliverable: `results/experiment_6641_operational_state_handoff.json`
- Gate output: `operational_state_ready_score`
- Research anchor: arXiv:2608.24569

### Exp6642 - Held-out live ARC actionability canary

Attach the qualified handoff to the real `E3AgentPolicy` path. Compare frozen,
context-only, and typed-memory arms on held-out, game-agnostic transitions. Log
retrieval, pre-generation routing, prompt influence, parsed action, and exact
environment result. This is a generalization and actionability test. It is not a
game or level solve task.

- Model: Qwen3.6-35B-A3B
- Deliverable: `results/experiment_6642_arc_live_actionability.json`
- Gate outputs: `arc_actionability_rows_ready_score`,
  `arc_action_influence_score`
- Prior scope: Exp6611

### Exp6643 - Exact component-patch admission

From Exp6642 failures, localize one game-agnostic memory, router, or feedback
component. Admit a patch only when an exact checker proves source repair,
held-anchor retention, support preservation, and rollback. Same-model
self-grades are diagnostic only.

- Deliverable: `results/experiment_6643_exact_component_patch_gate.json`
- Gate outputs: `patch_contract_ready_score`, `eligible_patch_count`
- Prior scope: Exp6614
- Research anchors: arXiv:2608.00017 and arXiv:2608.00220

### Exp6644 - Prospective multi-order continuous self-learning

Run frozen, context-only, and verified-memory arms prospectively across at least
three seeds and three task orders. Skills must record invocation and evidence and
may append, revise, or retire. Measure current, future, cross-constraint,
recoverable-support, regression, restart, and rollback effects. Update admission
comes only from exact outcomes.

- Model: Qwen3.6-35B-A3B
- Deliverable: `results/experiment_6644_prospective_skill_evolution.json`
- Gate output: `prospective_future_benefit_delta`
- Prior scope: Exp6614
- Research anchors: SkillForge, CAFE, ContinualSkillBench, and Recuris

### Exp6645 - Dense-family self-learning confirmation

If Exp6644 shows positive prospective future benefit, replay the frozen contract
with Gemma-4-31B. Keep event order, budgets, exact authority, and memory records
fixed. The result tests family transfer; it does not pool away per-order losses.

- Model: Gemma-4-31B
- Deliverable: `results/experiment_6645_dense_skill_evolution_confirmation.json`

### Exp6646 - Independent milestone capstone

Aggregate every terminal artifact, including blocks and nulls. Recompute gates
and headline values from rows. Separate measured facts, inferences, and open
questions. Reconcile the roadmap, architecture, traceability, status, changelog,
and completed-research records without changing the conductor or active roadmap.

- Deliverable: `results/experiment_6646_v579_capstone.json`
- Hardware: no new inference

## Dependency graph

```text
Exp6633 lease
  `-> Exp6634 model admission
        |-> Exp6635 direct headroom
        |     `-> Exp6636 delayed decoding
        |            `-> Exp6638 exact release audit
        |-> Exp6637 verifier unit calibration
        `-> Exp6641 operational handoff
              `-> Exp6642 ARC live actionability
                    `-> Exp6643 exact patch gate
                          `-> Exp6644 prospective CSL
                                `-> Exp6645 dense confirmation

Exp6639 Kac--Ward reference
  `-> Exp6640 local spectral replay

All terminal artifacts, including blocked ones
  `-> Exp6646 independent capstone
```

Exp6646 is deliberately ungated. It must report a blocked or null branch instead
of disappearing when an upstream gate fails.

## Gate contract

| Downstream | Upstream field | Condition |
|---|---|---|
| Exp6634 | Exp6633 `gpu_lease_scheduler_ready_score` | `== 1.0` |
| Exp6635 | Exp6634 Qwen and Gemma-26 admission scores | both `== 1.0` |
| Exp6636 | Exp6635 `headroom_ready_score` | `== 1.0` |
| Exp6637 | Exp6634 Qwen and Gemma-31 admission scores | both `== 1.0` |
| Exp6638 | Exp6636 `decoding_rows_complete_score` | `== 1.0` |
| Exp6640 | Exp6639 `kac_ward_reference_ready_score` | `== 1.0` |
| Exp6641 | Exp6634 `qwen36_admitted_score` | `== 1.0` |
| Exp6642 | Exp6641 `operational_state_ready_score` and Exp6634 Qwen admission | both `== 1.0` |
| Exp6643 | Exp6642 `arc_action_influence_score` | `> 0.0` |
| Exp6644 | Exp6643 patch readiness and eligible count | ready `== 1.0`, count `> 0` |
| Exp6645 | Exp6644 `prospective_future_benefit_delta` and Exp6634 Gemma-31 admission | delta `> 0.0`, admission `== 1.0` |

Every named field appears in the upstream task's required artifact fields with
the same spelling. All gates point backward within this roadmap.

## Hardware requirements

| Resource | Tasks | Requirement |
|---|---|---|
| CPU and RAM | All tasks | Schema checks, exact verifiers, Kac--Ward reference, reducers, and tests |
| Dual RTX 3090 | Exp6634-Exp6637, Exp6640-Exp6642, Exp6644-Exp6645 | Task-scoped leases, explicit device UUIDs, VRAM and unload receipts |
| Local GGUF cache | LLM tasks | Exact repo ID, file path, size, and SHA-256; no silent substitute |
| Rust toolchain | Exp6639-Exp6640 | Bounded reference and spectral implementation checks |
| KV260, GateMate, PolarFire | None | No changed-state receipt; no board task |
| Extropic TSU / Z1 | None | No authenticated runner; early access remains scheduled for 2027 |

Legacy Qwen3.5-0.8B and Gemma-4-E4B may run CPU smoke tests only. They cannot
produce headline rows or satisfy a model gate.

## Acceptance and reporting rules

- Every task writes one unique atomic JSON deliverable.
- Every task declares `verdict_class` from the closed enum.
- Every blocked artifact writes `gate_check_summary` with the failed check and
  observed value.
- Every comparative task emits per-unit rows. Aggregates cannot replace rows.
- Every LLM task declares `MODEL_SPECS` with at least one mandated model.
- ARC work targets the held-out live policy. It makes no level solve claim and
  does not use an offline game adapter as credit.
- Reruns cite prior honest verdicts, name the changed condition, and retire the
  scope if the same verdict returns.
- Exp6646 includes negative, blocked, disqualified, circular, and partial
  evidence in the capstone denominator.

## Deferred work

- Matching-base EBT or ARM-EBM comparisons remain deferred until public weights
  and a reproducible local runner exist.
- Kona remains a product comparator until public weights and a local method
  appear.
- TSU and attached-FPGA measurements remain deferred until a changed-state
  hardware receipt exists.
- KAN work remains deferred because it does not close one of the three current
  gaps.
- Game-level ARC claims remain deferred. This milestone improves live held-out
  policy behavior only.

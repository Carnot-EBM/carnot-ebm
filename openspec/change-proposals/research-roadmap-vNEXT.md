# Research Roadmap vNEXT - Milestone 2026.08.577

**Milestone:** `2026.08.577`
**Title:** Execution-Qualified Constraint Search and State-Grounded Self-Learning
**Status:** Proposed
**Planned experiments:** 13 across 4 phases
**Planning date:** 2026-08-26

## What milestone 2026.08.576 proved

Milestone 576 reached a terminal disposition for every activated task. Its most
important result was an evidence-boundary result: the execution substrate failed
before the flagship-model hypothesis could be tested.

1. **The exact benchmark exists.** Exp6604 produced 72 frozen two-level plan
   tasks with calibration and held splits, separate syntax and semantic
   compilers, and an independent exact executor. This is null infrastructure,
   not evidence that constrained decoding helps.
2. **None of the three flagship baselines became eligible.** Exp6605 blocked on
   Qwen process receipts, Exp6606 blocked on rerun-discipline metadata, and
   Exp6607 blocked on GPU ownership. Exp6608 therefore froze an empty eligible
   family set. Exp6609 and Exp6610 were correctly skipped before treatment and
   safety work consumed model time.
3. **The live ARC projection was reachable but inert.** Exp6611 traversed the
   actual `make_carnot_agent -> E3AgentPolicy` path. Selected, random, and
   no-projection arms had identical exact next-frame mismatch. No game or level
   solve was claimed.
4. **The Rust spectral sampler is promising but not yet claim-eligible.**
   Exp6612 retained 60 matched rows and reported large transition and wall-time
   gains over sequential Gibbs for the Rust arm. The artifact remained blocked
   because repository test and protection receipts failed. The result is CPU
   software only.
5. **Memory conformance did not become learning.** Exp6613 passed lifecycle,
   poison, restart, rollback, and quarantine checks without a utility claim.
   Exp6614 then found zero held-future benefit across 36 later events because the
   memory treatment never changed a prediction. Its invalid `blocked_tests`
   verdict class also disqualified the source artifact from a positive claim.
6. **The capstone correctly remained partial.** Exp6615 preserved the blocked
   decoding path, null ARC result, partial sampler evidence, lifecycle-only
   memory result, and zero-benefit continuous-learning result.

The honest V576 conclusion is that Carnot now has the right exact fixtures and
two promising mechanisms, but it does not yet have execution-qualified flagship
science or an active continuous-learning mechanism.

## The three biggest gaps to the PRD vision

### Gap 1: experiment execution is not a trustworthy scientific substrate

FR-12 and the model-tier requirements assume that the selected local model ran
under an owned accelerator process. V576 instead exposed wrong-model residency,
missing process receipts, non-owned GPU state, and a monolithic family reducer
with no complete rows. A result cannot be scientific when model identity and
accelerator ownership are unresolved.

V577 response:

1. Define task-scoped phase and accelerator receipts before changing the runner.
2. Add a readiness-aware GPU lease with model, device, PID, VRAM, phase, start,
   unload, and failure evidence.
3. Validate the lease with short independent canaries for all three mandated
   GGUF families.
4. Requalify only Qwen3.6 on a bounded frozen subset. Do not recreate the failed
   all-family prerequisite.
5. Let an independent no-model reducer freeze whether usable headroom exists.

### Gap 2: promising constraint and sampler mechanisms lack eligible evidence

The two-level decoder never ran, while the spectral sampler's rows were trapped
behind test and protection failures. The PRD vision requires both useful
verification and a path to efficient energy search, not merely implemented
interfaces.

V577 response:

1. Run direct, syntax-only, and two-level semantic decoding only after an
   independent headroom gate opens.
2. Keep the exact executor distinct from the decoding automata and audit false
   accepts across candidate budgets.
3. Repair spectral evidence integrity separately from the sampler algorithm.
4. Reproduce sampler scale gains in a fresh process with exact/reference parity,
   autocorrelation, setup, transition, ESS, and wall-time rows.
5. Keep every sampler statement software-only. Do not infer FPGA or TSU speed.

### Gap 3: continuous self-learning stores state but does not alter behavior

FR-11 requires autonomous directed self-learning with immutable validation,
chronology, retention, rollback, and timeouts. V576 satisfied much of the safety
contract, but the learned records did not influence future predictions. A
well-governed no-op is not learning.

V577 response:

1. Use Recuris-style verified working state to decide when and what memory is
   invoked.
2. Require a positive-control influence trace before prospective evaluation.
3. Localize each failure to a record, state field, invocation rule, or checker;
   patch only the implicated component.
4. Admit a patch only when it repairs the source failure and preserves held
   anchor tasks.
5. Evaluate multiple seeds and chronological plus shuffled task orders to expose
   variance and hidden curricula.
6. Keep generator weights frozen. Commit only exact-verified external state.
7. Include a held-out, game-agnostic ARC live-path test without solve credit.

## Research inputs added before planning

The V577 source refresh is recorded in `research-references.md`.

- **Recuris, arXiv:2608.24876** motivates verified working-state invocation,
  structured influence traces, component-scoped patching, and held-anchor
  admission gates.
- **On the Fragility of Self-Improving Agents, arXiv:2608.18066** requires
  multiple runs, shuffled-order controls, and explicit environment feedback.
  One chronological run is not sufficient evidence.
- **Meta-Ctrl, arXiv:2608.22149** remains the treatment design for separating
  token syntax from semantic plan constraints. V576 did not execute it.
- **Spectral Partitioning for k-Block Averaging Markov Chains,
  arXiv:2608.21466** remains the sampler method, but V577 first repairs the
  evidence contract and then performs independent replay.
- **Scaling Up Thermodynamic AI Models, arXiv:2607.00170**, and the FPGA Ising
  decomposition work, arXiv:2602.15985, motivate schedule, autocorrelation,
  setup, and communication accounting. They do not authorize hardware claims.
- Current OpenReview, Hugging Face, GitHub, Extropic, Kona, KAN, and hardware
  checks supplied no public checkpoint or authenticated runner that changes
  Carnot's exact-authority or hardware-access boundaries.

## Milestone thesis

The milestone asks three ordered questions:

1. Can Carnot prove that the intended model ran in an owned task-scoped GPU
   lease and produce a complete headroom-bearing baseline?
2. Given that prerequisite, do exact syntax and semantic constraints improve
   Qwen3.6 plan generation without increasing exact false accepts?
3. Can verified external memory demonstrably change future decisions and improve
   held outcomes across seeds and task orders without regression?

The spectral branch asks a parallel substrate question: does V576's Rust gain
survive a clean independent replay after evidence integrity is repaired?

## Target architecture

```text
                         EXECUTION CONTROL PLANE

 roadmap task -> phase receipt -> readiness-aware GPU lease -> model-bound PID
      |               |                    |                     |
      |               +-- start/end -------+-- VRAM/device ------+
      |                                                        unload
      v
 all-family short canaries -- per-family readiness --+
                                                     |
                                      Qwen-ready only v
 frozen Exp6604 subset -> direct Qwen rows -> independent headroom reducer
                                                     |
                                        headroom > 0 v

                         CONSTRAINT SEARCH PATH

 direct control <--- token syntax DFA <--- semantic action automaton
       \                    |                       /
        +------------- Qwen3.6 candidate search ---+
                              |
                              v
                    independent exact executor
                              |
                              v
                   row replay + false-accept audit

                      STATE-GROUNDED LEARNING PATH

 make_carnot_agent -> E3AgentPolicy -> verified working state
                                          |
                                          v
                  event-triggered constraint/skill invocation
                                          |
                                          v
                       candidate ranking or action selection
                                          |
                                          v
                       exact environment observation
                                          |
                                          v
 state + invoked item + action + observation + checker decision
                                          |
                                          v
          failure localization: record | state | router | checker
                                          |
                                          v
              component-scoped patch -> held-anchor gate
                                          |
                          accepted patch affects future events

                         SAMPLING SUBSTRATE

 Exp6612 retained rows -> evidence/test repair -> clean reference parity
             |                                         |
             +-- Python Gibbs / Rust Gibbs controls ---+
                                                       |
                                      independent scale replay
                                                       |
                         software schedule/cost receipt only

 Every path ends at exact checks and independently replayable rows.
 Learned memory, model scores, decoding automata, and sampler diagnostics do not
 grant release authority. No path in V577 claims a game solve or hardware speed.
```

## Phase I: execution truth and bounded headroom

### Exp6616 - V577 phase-receipt and retirement contract

Freeze the milestone task graph, prior-failure declarations, exact gate-field
owners, model policy, protected-file hashes, and a schema for phase and
accelerator receipts. Reconcile the actual V576 terminal evidence without
relabeling blocks as nulls.

**Acceptance:** every task, deliverable, gate owner, verdict class, prior failure,
model requirement, and protected file validates; `execution_contract_ready_score`
is `1.0` only for a replayable contract. This is null infrastructure.

**Deliverable:** `results/experiment_6616_v577_execution_contract.json`

### Exp6617 - readiness-aware GPU lease and phase receipts

Implement a reusable task-scoped lease around the existing local llama.cpp CUDA
path. Bind a task ID, model identity, device, PID, process start, VRAM delta,
phase transitions, heartbeat, timeout, unload, and terminal failure to one
atomic receipt. Use mocks and process fixtures; do not run a model.

**Acceptance:** concurrent claims on one device fail closed; stale leases are
diagnosed but not silently stolen; wrong model, PID reuse, missing unload, and
receipt tampering fail; `gpu_lease_scheduler_ready_score=1.0` only when focused
and repository protection tests pass.

**Deliverable:** `results/experiment_6617_gpu_lease_phase_receipts.json`

### Exp6618 - mandated-model accelerator admission canary

Use the lease to run one bounded fresh-process canary for Qwen3.6 35B-A3B,
Gemma 4 31B, and Gemma 4 26B-A4B. Each family gets an independent readiness
field, so one failure does not recreate a monolithic family gate.

**Acceptance:** each ready family has exact GGUF, tokenizer, chat template,
device, PID, VRAM, CUDA-offload, raw-output, timing, and unload evidence.
`qwen_admission_ready_score` is owned here and gates later Qwen work.

**Deliverable:** `results/experiment_6618_mandated_model_admission.json`

### Exp6619 - bounded Qwen3.6 direct headroom requalification

Run the frozen direct prompt on a preregistered 12-calibration/12-held subset of
Exp6604 with two seeds. Use Qwen3.6 only, preserve every failure, and do not run a
constraint treatment.

**Acceptance:** all 48 rows replay from raw output through the independent exact
executor; `qwen_headroom_rows_ready_score=1.0` records complete evidence;
`qwen_headroom_ready_score=1.0` only when held direct exact success is in the
preregistered 20%-80% interval.

**Deliverable:** `results/experiment_6619_qwen36_bounded_headroom.json`

### Exp6620 - independent bounded-headroom reducer

Replay Exp6619 without a model. Validate lease ownership, model identity, raw
rows, failures, subset hashes, exact outcomes, and aggregate agreement. Freeze
eligibility before a treatment runs.

**Acceptance:** blocked or incomplete source evidence remains blocked;
`v577_headroom_ready_score=1.0` only for independently complete Qwen rows with
held exact success in the 20%-80% interval.

**Deliverable:** `results/experiment_6620_headroom_reducer.json`

## Phase II: two-level constraint search and safety

### Exp6621 - headroom-gated two-level constrained decoding

When Exp6620 opens the gate, compare direct, syntax-only, and syntax-plus-semantic
search on the same frozen held tasks and seeds with Qwen3.6. Preserve direct rows
unchanged, charge every generation and failure, and use the independent executor
as final authority.

**Acceptance:** paired per-unit rows reproduce; direct rows match Exp6619;
automata cannot certify themselves; `decoding_rows_ready_score=1.0` records a
complete comparison. Positive science additionally requires a preregistered
paired exact-success gain, no false-accept increase, and cost accounting.

**Deliverable:** `results/experiment_6621_two_level_decoding.json`

### Exp6622 - independent decoding authority and safety audit

Replay Exp6621 without a model. Recompute paired effects, syntax and semantic
violations, exact false accepts, candidate-budget curves, and total charged cost.
Attack omitted obligations and contaminated learned feasible sets.

**Acceptance:** every unit and budget recomputes; exact execution remains release
authority; any oracle-defined or circular result is classified accordingly.

**Deliverable:** `results/experiment_6622_decoding_safety_audit.json`

## Phase III: spectral sampler evidence recovery

### Exp6623 - spectral sampler integrity and reference repair

Repair the failed repository-test and protected-file evidence around Exp6612.
Build a clean exact-enumerable reference suite and independently bind Python and
Rust implementations, seeds, compiler flags, binaries, and stationary checks.

**Acceptance:** clean tests and protection checks pass; Python and Rust Gibbs
controls match the reference; spectral partitions are deterministic;
`sampler_integrity_ready_score=1.0` is a null readiness result.

**Deliverable:** `results/experiment_6623_spectral_integrity_repair.json`

### Exp6624 - independent spectral scale replay and cost envelope

In a fresh process, replay sequential Gibbs, Python spectral, and Rust spectral
arms on preregistered frustrated `n=16`, `n=32`, and `n=64` fixtures. Report setup,
transitions, autocorrelation, ESS, wall time, and charged end-to-end cost.

**Acceptance:** exact/reference quality is noninferior, rows are complete, binary
identity is bound, and any gain survives bootstrap intervals and setup charges.
All conclusions say CPU software only. FPGA and TSU values are analytical
portability descriptors, not performance evidence.

**Deliverable:** `results/experiment_6624_spectral_scale_replay.json`

## Phase IV: live actionability and continuous self-learning

### Exp6625 - held-out ARC live-path memory actionability canary

Add a game-agnostic verified working-state and event-triggered constraint
invocation to the actual `E3AgentPolicy` path. Use disjoint development and
held-game archives. Compare static, invoked, and matched sham arms on exact
next-action or next-frame criteria. This is an actionability test, not a solve.

**Acceptance:** the scored import closure reaches the new path; a seeded positive
control changes ranking or action; held rows remain game-blind and row-complete;
`live_memory_activation_ready_score=1.0` requires observable influence, not
outcome benefit. No game or level solve is claimed.

**Deliverable:** `results/experiment_6625_arc_live_memory_actionability.json`

### Exp6626 - verified working-memory patch gate

Implement structured traces and component-scoped patching for external memory.
Localize failures to the constraint record, working-state field, invocation
policy, or checker. Admit a patch only when it repairs its source event and does
not regress held anchors. Preserve lifecycle, poison, restart, and rollback.

**Acceptance:** localization fixtures cover all four components; whole-memory
rewrites fail closed; accepted and rejected patches replay;
`memory_patch_contract_ready_score=1.0` requires activation, exact repair,
anchor retention, and recovery equality. This remains null infrastructure.

**Deliverable:** `results/experiment_6626_working_memory_patch_gate.json`

### Exp6627 - prospective multi-order continuous self-learning

Use a frozen Qwen3.6 candidate pool and the verified external memory-control
layer. Predict or select before observing each outcome, then localize and gate a
component patch for later events. Compare no-update, static memory,
state-grounded learning, and shuffled-update controls across at least three
seeds and three preregistered task orders.

**Acceptance:** generator hashes never change; treatment activation is nonzero;
held-future exact success improves over static and shuffled controls with a
paired interval excluding zero; held anchors, recoverable support, poison,
restart, rollback, and bounded occupancy pass. Otherwise the honest result is
null, partial, blocked, or disqualified.

**Deliverable:** `results/experiment_6627_prospective_state_grounded_learning.json`

### Exp6628 - V577 independent capstone and architecture reconciliation

Always run. Replay every available task and gate, preserve missing and blocked
branches, recompute comparative claims from rows, and reconcile relevant specs,
traceability, architecture, status, and changelog to terminal evidence.

**Acceptance:** all 12 upstream task dispositions are explicit; gate fields bind
their owner declarations exactly; no block becomes null, no software result
becomes hardware, no archive replay becomes an ARC solve, and FR-11/FR-12 move
only on eligible evidence. The capstone is `null` or `partial`, never positive.

**Deliverable:** `results/experiment_6628_v577_independent_capstone.json`

## Dependency graph

```text
Exp6616 execution contract
   |
   +--> Exp6617 GPU lease + phase receipts
   |       |
   |       +--> Exp6618 mandated-model admission
   |               |
   |               +--> Exp6619 bounded Qwen baseline
   |               |       |
   |               |       +--> Exp6620 independent reducer
   |               |               |
   |               |               +-- gate: headroom == 1 --> Exp6621 decoder
   |               |                                             |
   |               |                                             +--> Exp6622 audit
   |               |
   |               +-------------------------------------------> Exp6627 learning
   |
   +--> Exp6623 sampler integrity
   |       |
   |       +-- gate: integrity == 1 --> Exp6624 scale replay
   |
   +--> Exp6625 ARC live actionability
           |
           +-- gate: activation == 1 --> Exp6626 patch gate
                                               |
                                               +--> Exp6627 learning

Exp6628 capstone reads every available Exp6616-Exp6627 artifact and always runs.
```

Structured gates use fields declared by their owner tasks:

| Downstream | Upstream field | Condition |
|---|---|---|
| Exp6617 | `exp6616.execution_contract_ready_score` | `== 1.0` |
| Exp6618 | `exp6617.gpu_lease_scheduler_ready_score` | `== 1.0` |
| Exp6619 | `exp6618.qwen_admission_ready_score` | `== 1.0` |
| Exp6621 | `exp6620.v577_headroom_ready_score` | `== 1.0` |
| Exp6622 | `exp6621.decoding_rows_ready_score` | `== 1.0` |
| Exp6624 | `exp6623.sampler_integrity_ready_score` | `== 1.0` |
| Exp6626 | `exp6625.live_memory_activation_ready_score` | `== 1.0` |
| Exp6627 | `exp6618.qwen_admission_ready_score` and `exp6626.memory_patch_contract_ready_score` | both `== 1.0` |

## Hardware requirements and boundaries

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| Dual RTX 3090 CUDA GPUs | Exp6618, Exp6619, Exp6621, Exp6627 | Use task-scoped leases. One task owns a device and model process at a time. Record exact GGUF identity, PID, VRAM, offload, heartbeat, unload, and failures. No silent CPU fallback in headline rows. |
| Local GGUF cache | Exp6618, Exp6619, Exp6621, Exp6627 | Required models are `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`. Verify shards, tokenizer, and chat template before inference. |
| CPU and Rust toolchain | Exp6617, Exp6620, Exp6622-Exp6626, Exp6628 | Needed for exact execution, independent replay, sampler builds, tests, and audits. Exp6624 reports CPU software only. |
| RAM and disk | all tasks | Retain raw responses, model and process receipts, checkpoints, per-unit rows, Rust binaries, fixture hashes, and atomic artifacts. Check capacity before compute. |
| KV260, GateMate, PolarFire | none scheduled for execution | Preserve existing changed-state-only continuity. Do not repeat unchanged board probes. No V577 board latency or speed claim. |
| Extropic XTR-0/Z1 | none | No authenticated runner. Z1 early access remains planned for 2027. No TSU execution, power, latency, or availability claim. |

## Model policy

Every task that performs LLM inference includes at least one mandated model in
its `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6618 uses all three. Exp6619, Exp6621, and Exp6627 use Qwen3.6. Legacy
Qwen3.5-0.8B and Gemma E4B models may appear only as labeled CPU smoke tests;
they cannot supply a headline row, replace a failed flagship run, or satisfy a
gate.

## Claim and safety boundaries

1. Exact deterministic checks remain release authority. A learned score,
   memory gate, decoding automaton, or sampler diagnostic cannot certify itself.
2. Infrastructure readiness uses `verdict_class=null`. It is not positive
   science.
3. Every comparative task emits per-unit rows. Aggregates are recomputed from
   those rows and preserve failures.
4. Every genuine blocked verdict begins with `blocked_*` and includes the exact
   `gate_check_summary` field with observed values.
5. Oracle-defined or self-certified gains cannot use `verdict_class=positive`.
6. Exp6625 makes no game or level solve claim. It does not read game source,
   build per-game adapters, run offline ground-truth BFS, or update the ARC solve
   registry.
7. Exp6624 is a CPU software result. Analytical FPGA or TSU descriptors are not
   measured hardware evidence.
8. Exp6627 keeps model weights and the outer improvement procedure immutable.
   Only typed external state can change, and every accepted patch is journaled,
   reversible, and exact-gated.
9. A self-learning claim requires nonzero treatment activation, prospective
   held-future benefit, multi-order robustness, retention, support, poison,
   restart, rollback, and bounded occupancy.
10. `research-roadmap.yaml` and `scripts/research_conductor.py` are protected and
    must remain byte-identical throughout the milestone.

## Decentralization implications

V577 strengthens the boundary between proposal and authority. Local models can
generate plans and a local memory layer can adapt routing, but every admission,
release, and scientific claim remains reconstructible from local immutable
evidence. Task-scoped GPU leases prevent one worker's model process from
silently becoming another worker's evidence. Component-scoped memory patches
avoid an opaque central rewrite and keep each change attributable to a verified
failure. The resulting artifacts can be replayed by an independent node without
trusting the original model process, reducer, or planner.

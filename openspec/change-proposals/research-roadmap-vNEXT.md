# Research Roadmap vNEXT - Milestone 2026.08.576

**Milestone:** `2026.08.576`
**Title:** Verifier-Governed Constraint Decoding and Reversible Live Learning
**Status:** Proposed
**Planned experiments:** 12 across 4 phases
**Planning date:** 2026-08-25

## What milestone 2026.08.575 proved

Milestone 575 completed the six tasks in the active roadmap. The longer draft
roadmap listed later memory and hardware tasks, but those tasks were not in the
activated YAML and did not run.

1. **CFR had no measurable opportunity.** Exp6593 replayed the Qwen3.6 and
   Gemma 4 31B direct and constraint-first rows. Every exact-success delta was
   `0.0` because the direct arms already had 100% exact success. Exp6594 found a
   clean authority chain but no CFR effect. Exp6593's stored artifact is
   adversarial-flagged for its original duration receipt, so its null science
   result is useful as a planning constraint, not a clean headline.
2. **Invariant projection worked on its bounded target.** Exp6595 found that a
   held selected invariant reduced rollout error on frozen conservative
   fixtures. The random control was worse and damped dynamics removed the
   effect. This is positive local mechanism evidence, not a live ARC result.
3. **Feasible-token flow remained a toy result.** Exp6596 proved its local
   convex-hull and endpoint checks on fixed embeddings. Its result is
   `circular_positive` and does not establish language-model training or useful
   decoding.
4. **Spectral block sampling improved transition efficiency.** Exp6597 found
   stationary-distribution noninferiority and effective-sample-size gains per
   transition on exact-enumerable six-spin fixtures. Charged wall time was not
   better. No Rust, FPGA, TSU, or larger frustrated-system claim exists.
5. **Continuous self-learning remains unproved.** No active V575 task executed
   the planned Gene or prospective learning chain. Earlier chronological
   learning either had no held-future benefit or blocked before live evidence.

The honest V575 conclusion is narrow: Carnot has two promising local
mechanisms, but no headroom-bearing flagship decoding result, no reachable
live-world-model projection result, and no eligible continuous self-learning
benefit.

## The three biggest gaps to the PRD vision

### Gap 1: exact constraints are not yet improving flagship generation

FR-12 calls for verifiable reasoning, but the latest flagship comparison had
zero baseline headroom. Carnot cannot learn anything about constrained decoding
from a treatment applied to rows that are already all correct.

V576 response:

1. Build a fixed exact-executable plan corpus with calibration and held splits.
2. Separate token syntax from action-level preconditions, goals, and ordering.
3. Qualify headroom independently for Qwen3.6 35B-A3B, Gemma 4 31B, and Gemma 4
   26B-A4B.
4. Compare direct, syntax-only, and two-level constrained decoding only on
   family-level held splits with preregistered headroom.
5. Keep the exact simulator outside the decoding automata so an incomplete
   encoding cannot certify itself.

### Gap 2: positive energy mechanisms are not reusable runtime primitives

Exp6595 and Exp6597 are experiment modules. The invariant result is not reachable
from `E3AgentPolicy`. The sampler result is limited to tiny Python fixtures and
loses on wall time. This leaves the PRD's live reasoning and hardware paths
unserved.

V576 response:

1. Move invariant projection into a default-off, game-agnostic live
   world-model path.
2. Replay archived live-agent world-model transitions through the actual
   `E3AgentPolicy` import closure. Make no game or level solve claim.
3. Move spectral k-block sampling into the reusable sampler interface, scale to
   frustrated `n=16` and `n=32` systems, and add Rust/Python parity.
4. Separate setup cost, transitions, effective sample size, and wall time.
5. Stop at a software portability receipt. Attached boards remain
   changed-state-only, and Extropic remains unavailable.

### Gap 3: continuous self-learning has no prospective held-future win

FR-11 requires autonomous directed self-learning with immutable validation,
rollback, and timeouts. Exp6496 was row-complete but failed held-future benefit.
Exp6552 found no benefit from conflict hysteresis. Exp6553 blocked on the GPU
and restart contracts before a prospective SOTA result existed.

V576 response:

1. Persist exact verifier evidence with every candidate invariant record.
2. Use provisional, active, quarantined, and archived lifecycle states.
3. Keep world-model and generator weights immutable.
4. Predict before observing each next frame, then use the observed frame for an
   exact post-event admission decision.
5. Compare no learning, a static projector, verifier-governed online memory,
   and a matched shuffled control in chronological order.
6. Require held-future benefit, retention, recoverable support, poison safety,
   restart equality, and exact rollback before any default-off shadow consumer
   can be considered.

## Research inputs added before planning

The V576 source refresh is in `research-references.md`.

- **MemGuard, arXiv:2608.21867** motivates persistent verifier metadata and
  explicit memory lifecycle states. Carnot will reproduce the contract, not
  import the paper's benchmark claims.
- **Safety Hacking, arXiv:2608.22915** requires false-accept and
  candidate-budget curves. A learned feasible-set filter cannot be release
  authority.
- **Meta-Ctrl, arXiv:2608.22149** motivates the token/action factorization in
  the headroom benchmark and decoder.
- **Prime Agent, arXiv:2608.23552** motivates immutable base policy,
  evidence-backed side-state changes, snapshots, and rollback.
- **Verifier-Induced Support Reshaping, arXiv:2608.00220** remains the reason to
  measure future recoverable support, not only immediate success.
- Current OpenReview, Hugging Face, GitHub, Extropic, and Kona checks do not
  change exact authority or provide an authenticated new hardware runner.

## Milestone thesis

The milestone asks two linked questions:

1. Can exact constraints improve local flagship generation when a frozen held
   benchmark has real headroom?
2. Can a positive energy correction become a reversible live-path learning
   primitive without losing safety, retention, or future support?

The sampler work is a parallel substrate question: can Exp6597's algorithmic
gain survive scale and a reusable Rust boundary before hardware is discussed?

## Target architecture

```text
                     EXACT-CONSTRAINT GENERATION PATH

 fixed plan generator + exact simulator
                 |
                 v
 calibration split + frozen held split
                 |
       +---------+----------+----------------+
       |                    |                |
       v                    v                v
 Qwen3.6 direct       Gemma 4 31B      Gemma 4 26B-A4B
       |                    |                |
       +--------- immutable baseline rows --+
                              |
                              v
                     independent headroom gate
                              |
                              v
       direct control <-- token DFA <-- action semantic automaton
              \               |               /
               +---------- local GGUF --------+
                              |
                              v
                     independent exact simulator
                              |
                              v
                 row replay + safety-hacking audit

                        LIVE LEARNING PATH

 make_carnot_agent -> E3AgentPolicy -> executable world-model proposal
                                         |
                                         v
                          default-off invariant projector
                                         |
                           +-------------+-------------+
                           |                           |
                           v                           v
                    predicted next frame       exact observed frame
                           |                           |
                           +-------- exact error ------+
                                         |
                                         v
                verifier-governed invariant lifecycle memory
                   provisional -> active -> quarantine/archive
                                         |
                              chronological retrieval
                                         |
                                         +----> next prediction

                         SAMPLING SUBSTRATE

 Exp6597 spectral partition -> reusable Python sampler -> Rust parity
             |                         |                    |
             +-- Gibbs controls -------+---- n=16/n=32 -----+
                                                      |
                                      future FPGA/TSU cost receipt only

 All three paths end at exact checks and independent row replay.
 Learned scores, automata, projectors, and memory records never release output.
```

## Phase I: exact headroom and family qualification

### Exp6604 - exact two-level plan corpus and compiler contract

Create a deterministic plan-task generator with calibration and held splits.
Each task has a grounded action vocabulary, token grammar, action-level
preconditions, ordering rules, goals, and an independent exact executor. Add a
two-level compiler that factors token syntax from semantic action state. Create
mutations that pass one layer but fail the exact executor.

**Acceptance:** all fixtures and split hashes are deterministic; the exact
executor has hand-checked and mutation-tested authority; at least one
automaton-accepted candidate is rejected by exact execution; no LLM runs; and
`headroom_fixture_ready_score=1.0` only when all contracts replay.

**Deliverable:**
`results/experiment_6604_exact_two_level_plan_corpus.json`

### Exp6605 - Qwen3.6 direct baseline headroom

Run the frozen direct prompt on
`unsloth/Qwen3.6-35B-A3B-GGUF`. Preserve raw bytes, exact model identity,
embedded GGUF chat-template and tokenizer receipts, CUDA offload evidence,
seeds, failures, tokens, and exact results for every calibration and held unit.

**Acceptance:** every expected row is present and exact-replayable. This task
does not need to find headroom to complete. It sets
`qwen_headroom_ready_score=1.0` only when held direct exact success is inside the
preregistered 20%-80% interval.

**Deliverable:** `results/experiment_6605_qwen36_direct_headroom.json`

### Exp6606 - Gemma 4 31B direct baseline headroom

Repeat the byte-frozen baseline independently with
`unsloth/gemma-4-31B-it-GGUF`. Do not read Qwen outputs or tune task wording from
another family.

**Acceptance:** same row, identity, failure, and exact-replay gates as Exp6605.
Set `gemma31_headroom_ready_score=1.0` only for held direct exact success in the
20%-80% interval.

**Deliverable:** `results/experiment_6606_gemma4_31b_direct_headroom.json`

### Exp6607 - Gemma 4 26B-A4B direct baseline headroom

Repeat the byte-frozen baseline independently with
`unsloth/gemma-4-26B-A4B-it-GGUF`. This is a headline-capable middle MoE, not a
legacy smoke model.

**Acceptance:** same row, identity, failure, and exact-replay gates as Exp6605.
Set `gemma26_headroom_ready_score=1.0` only for held direct exact success in the
20%-80% interval.

**Deliverable:** `results/experiment_6607_gemma4_26b_direct_headroom.json`

## Phase II: two-level decoding and independent safety

### Exp6608 - independent family headroom reducer

Replay Exp6605-Exp6607 without loading a model. Fix the eligible family list
before any constrained treatment runs. Preserve incomplete or blocked families
instead of making eligibility depend on treatment output.

**Acceptance:** at least one complete family has held direct exact success in
the 20%-80% interval; all eligible family and unit hashes are frozen; no held
unit is selected by its individual outcome; and
`headroom_benchmark_ready_score=1.0` only under those conditions.

**Deliverable:** `results/experiment_6608_family_headroom_reducer.json`

### Exp6609 - local GGUF two-level constrained decoding comparison

For every eligible family, reuse the direct rows and run syntax-only and
two-level syntax-plus-semantic decoding on the same held task bytes. Use the
existing llama.cpp GGUF loader and reviewed grammar adapters. Preserve reasoning
before the structured plan boundary. Do not use a prompt-only fallback as the
constrained treatment.

**Acceptance:** the treatment is a real token-mask/action-state intervention;
all arms are dose- and seed-accounted; exact execution decides success; and a
candidate win requires a positive paired exact-success delta, a nonnegative
paired lower bound, no unsafe-release increase, and the frozen cost limit. Any
eligible win is `circular_positive` because exact constraints define the target.
`two_level_decoder_comparison_ready_score` reports row-complete comparison, not
benefit.

**Deliverable:**
`results/experiment_6609_two_level_constrained_decoding.json`

### Exp6610 - independent safety-hacking and authority audit

Replay Exp6609 from raw rows. Vary candidate budget without new model inference
when stored candidates suffice. Inject high-score candidates that pass an
incomplete automaton but fail the exact executor. Test grammar removal,
semantic-state corruption, family swaps, source tamper, and exact-authority
substitution.

**Acceptance:** false accepts never release; contamination and false-accept
curves are reported by candidate budget; every source and authority mutation
fails closed; and `constraint_safety_audit_ready_score=1.0` only when all
eligible rows and attacks replay.

**Deliverable:**
`results/experiment_6610_constraint_safety_hacking_audit.json`

## Phase III: live projection and scalable sampling

### Exp6611 - live-path invariant projection replay

Extract the bounded projector from Exp6595 into a reusable, default-off module.
Wire it through `make_carnot_agent` and `E3AgentPolicy` so the scored path can
reach it without a per-game adapter. Replay immutable archived live-agent
world-model transitions with game-identity-blind features. Compare no
projection, selected invariant projection, and norm-matched random projection.

**Acceptance:** the actual live import closure reaches the projector; calibration
and held games are disjoint; no source code, outer-loop search, or offline
ground-truth solver is used; exact observed next frames decide prediction error;
and a candidate win requires lower held error, random-control separation, no
runtime-validity loss, and bounded cost. This task makes no solve, level, or
leaderboard claim.

**Deliverable:**
`results/experiment_6611_live_arc_invariant_projection.json`

### Exp6612 - frustrated spectral k-block scale and Rust parity

Move Exp6597's spectral partition sampler into the reusable sampler interface.
Compare sequential Gibbs, random k-block, and spectral k-block arms on fixed
frustrated `n=16` and `n=32` systems. Add Rust/Python parity and a software-only
hardware cost envelope.

**Acceptance:** reference moments or independent long-chain intervals are
declared before comparison; stationary quality is noninferior; parity matches
within tolerance; setup, transition, and wall costs are separate; and a win
requires either charged wall-time or ESS-per-wall-time gain. This is not
Spectral Annealing, PIMI, FPGA execution, or a TSU claim.

**Deliverable:**
`results/experiment_6612_spectral_k_block_scale_rust_parity.json`

## Phase IV: verifier-governed continuous self-learning

### Exp6613 - invariant-memory lifecycle conformance

Implement typed invariant records with exact verifier descriptors, source and
model hashes, uncertainty, lifecycle state, occupancy bounds, conflict rules,
restart journals, and rollback snapshots. Memory records are data, never
instructions. Revalidation occurs at admission and retrieval.

**Acceptance:** unsupported, stale, command-bearing, duplicated, poisoned, and
conflicting records cannot become active; restart and rollback are byte-equal;
the base policy and model weights remain immutable; the compact projection and
lookup arithmetic has a Rust/CPU hardware path; and
`invariant_memory_ready_score=1.0` only when every lifecycle mutation closes.

**Deliverable:**
`results/experiment_6613_invariant_memory_lifecycle.json`

### Exp6614 - prospective chronological invariant self-learning

This is the milestone's mandatory continuous self-learning task. Process
held live world-model transitions in chronological order. Each arm predicts
before the next frame is revealed. Compare no learning, the frozen static
projector, verifier-governed online memory, and a matched shuffled-admission
control. Commit side-state only after exact post-event verification.

**Acceptance:** generator and world-model hashes stay fixed; every opportunity
has every arm; no held-future or game-identity leakage occurs; unsafe commits
are zero; restart and rollback replay exactly; retention and future recoverable
support are noninferior; and the online arm has preregistered positive
held-future benefit over both static and shuffled controls.
`continuous_self_learning_task=true`. A win is `circular_positive` because the
exact observed frame governs admission and evaluation.

**Deliverable:**
`results/experiment_6614_prospective_invariant_self_learning.json`

### Exp6615 - independent milestone audit and architecture reconciliation

Always run. Recompute the decoding, safety, projection, sampler, and
self-learning claims from per-unit rows. Test gate-field spelling, missing and
blocked inputs, exact authority, model identity, chronology, rollback, support,
and non-claims. Reconcile capability specs, traceability, status, changelog, and
architecture only to evidence that exists.

**Acceptance:** every result has an independent disposition of positive,
circular-positive, null, blocked, disqualified, or partial; no blocked branch is
reported as null; no toy, archive replay, or software parity result becomes a
live or hardware claim; and all changed specs and operations documents agree.

**Deliverable:** `results/experiment_6615_v576_independent_capstone.json`

## Dependency graph

```text
Exp6604 exact corpus/compiler
   +--> Exp6605 Qwen baseline -------+
   +--> Exp6606 Gemma 31B baseline --+--> Exp6608 headroom reducer
   +--> Exp6607 Gemma 26B baseline --+             |
                                                   v
                                      Exp6609 two-level decoding
                                                   |
                                                   v
                                      Exp6610 safety/authority audit

Exp6595 positive local projection --> Exp6611 live-path projection
                                                   |
                                                   +--> Exp6613 memory contract
                                                            |
                                                            v
                                             Exp6614 continuous self-learning

Exp6597 positive tiny sampler ------> Exp6612 scale + Rust parity

Exp6604-Exp6614 ----------------------------------> Exp6615 capstone
```

Structured conductor gates exist only between tasks in the V576 YAML. Prior
milestone artifacts are immutable preconditions, not `gated_on` targets.
Independent reducers and the capstone always preserve missing or blocked
inputs.

## Hardware requirements

### Required and available

- **Two RTX 3090 GPUs:** use the reviewed ownership and reaper protections.
  Run one family per task. Record per-device free VRAM, process ownership,
  loaded model hash, CUDA layer offload, tokens per second, and unload receipt.
- **Cached GGUFs:**
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. Resolve through `cached_sota_pair()` and
  content-derived cache identity. Do not download during a task.
- **CPU and RAM:** exact plan execution, automata, archive replay, lifecycle
  tests, and independent reducers are CPU paths.
- **Rust toolchain:** required for sampler parity and the compact
  projection/memory portability receipt.

### Optional and explicitly out of scope

- **KV260:** no command is scheduled without a changed physical or bitstream
  state. A later task may map the proven sampler kernel through authenticated
  SSH, `xmutil`, and UIO receipts.
- **GateMate and PolarFire:** no repeated detection or unchanged continuity
  command. Preserve their existing terminal state.
- **Extropic X0/Z1:** Carnot has no authenticated runner. Torx and
  Thermalizers remain future compiler references. No latency, power, or
  availability claim is allowed.

## Model policy

Every live LLM task names at least one mandated headline GGUF in `MODEL_SPECS`.
Exp6605, Exp6606, and Exp6607 each isolate one family. Exp6609 may load only
families frozen as eligible by Exp6608, but its declared model registry includes
all three mandated IDs. The embedded GGUF tokenizer and chat template are part
of model identity. Legacy Qwen3.5-0.8B and Gemma 4 E4B may be used only for a
clearly labeled CPU smoke test and can never supply a milestone result.

## Claim boundaries

- The exact executor is oracle for the plan benchmark. Any decoding win is
  `circular_positive`, not a general positive verification claim.
- The live projection and self-learning tasks do not claim an ARC game solve,
  level solve, or leaderboard gain.
- The sampler task does not reopen Spectral Annealing or PIMI and makes no
  attached-hardware claim.
- A complete infrastructure task normally has `verdict_class=null`.
- Every comparative task emits per-unit rows. Every blocked verdict emits
  `gate_check_summary` with the failed check and observed value.
- No task edits `research-roadmap.yaml` or
  `scripts/research_conductor.py`, and no task pushes.

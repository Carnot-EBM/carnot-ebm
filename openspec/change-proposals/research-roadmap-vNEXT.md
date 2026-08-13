# Research Roadmap V548: Observable Proposals, Certified Rollback, and Falsifiable Goals

**Milestone:** `2026.08.548`
**Date:** 2026-08-12
**Status:** Proposed
**Task range:** Exp6363-Exp6376
**Execution file:** `research-roadmap-next.yaml`

## Milestone thesis

V547 reached the local GGUF runtime but did not produce one model token. All
three child processes exited with code 1. Each raw output was empty. The result
discarded the child stderr that could explain the failure. It also recorded
`n_ctx=512`, while the committed source now declares `n_ctx=2048`. The V547
authenticity result therefore proves model resolution and GPU availability. It
does not prove live proposal generation.

V548 fixes observability before it asks another scientific question. It binds
each model call to its source revision, prompt token count, context capacity,
child stderr, exit state, GPU samples, phase timing, and raw bytes. A corrected
three-model canary must pass before factor-learning tasks run.

The learning phase then uses a verifier-guided hypothesis frontier. It keeps
the strongest exactly verified factor and gives later proposal rounds only the
remaining immutable counterexamples. A chronological release stream keeps the
active factor read-only during proposal generation. An anytime e-process and
exact checks control each release. A dependency graph supports selective
rollback when a released factor becomes stale or poisoned.

The ARC phase completes the goal-evidence work that the seven-task V547 active
queue never activated. It adds a two-sided evidence contract, active reward
machine discrimination, evidence-response calibration, and a default-off live
shadow. The live agent sees only its own frames, actions, and legal action set.
The phase makes no game or level solve claim.

## What V547 proved

| Evidence | Result | V548 consequence |
|---|---|---|
| Exp6350 terminal handoff | The artifact preserved V546 boundaries, but the conductor quarantined it for short duration and missing method evidence | Rebuild the handoff from primary artifacts and preserve the flag. Do not use Exp6350 as clean evidence. |
| Exp6351 source freeze | `complete_null`; no strict post-marker source changed V547 scope | Open the V548 marker and freeze only the changed mechanisms. |
| Exp6352 authenticity preflight | Model files, embedded tokenizers, two RTX 3090 GPUs, and disk checks passed | Reuse model resolution and exact fixtures. |
| Exp6352 child calls | All three return codes were 1; raw bytes and token counts were zero; `models_used=[]`; `live_autoregressive_generation_invoked=false` | Diagnose the child process before any quality or learning claim. |
| Exp6352 evidence loss | The artifact omitted stderr and a top-level random seed; its recorded `n_ctx` differs from committed source | Add source-artifact drift checks, retained diagnostics, and capacity math. |
| Exp6353 proposal A/B | Gate-blocked three times on Exp6352 readiness 0 | Do not require the retired Exp6353 chain. Create a new chain after a changed runtime prerequisite. |
| Exp6354 prospective learning | Pre-emptively skipped because Exp6353 retired | Retry only after the new verified-frontier A/B produces positive live evidence. |
| Exp6355 future consumer | Gate-blocked on the missing Exp6354 deliverable | Keep the consumer default-off and gate it on both learning and rollback readiness. |
| Exp6356 safety audit | `complete_null`; it found missing, blocked, null, or unauthentic evidence and promoted no utility | Preserve its attacks. Add decomposition conflict and dependency rollback attacks. |
| V547 operational retro | It could not attribute model phase timing, GPU use, model count, or dispatcher choice | Make these receipts required in the new runtime contract. |
| V547 active queue | The active roadmap contained seven tasks, all now terminal | The earlier proposal-only ARC tasks were not experiments. Use new IDs and execute the work now. |

## The three largest gaps

### Gap 1: Local inference is not observable enough to support a claim

The runtime resolved all mandatory models and authenticated both GPUs. It then
lost the reason that every child failed. The artifact says generation ran in
its prose, but its boolean and process receipts say no generation occurred.
This is a measurement-contract failure before it is a model-quality failure.

V548 adds an observable child contract. It retains bounded stderr, hashes the
exact source and command, counts prompt tokens with the embedded GGUF
tokenizer, proves `prompt + output <= n_ctx`, samples task-linked GPU state,
and records load, prompt, generation, unload, and cleanup phases. The contract
fails closed on drift, empty output, nonzero exit, or missing telemetry.

### Gap 2: FR-11 still has no prospective live-model value

V546 proved factor release, retention, rollback, and bounded growth in replay.
V547 did not reach a live proposal. It therefore did not show that a real local
model can improve a factor or help a future consumer.

V548 uses all three mandatory local GGUF families. It compares independent
sampling with a verified-incumbent frontier at matched calls, tokens, exact
checks, and time. A later chronological stream seals future events before
learning. A release may affect only future events. A default-off consumer runs
only after exact release and dependency rollback both pass.

### Gap 3: ARC goal admission remains one-sided

Exp6258 accepted 21 false goals and only five true goals among 29 measurable
predicates. Acceptance precision was 0.1923. The active V547 queue did not run
the planned fix. The current gate can still accept a predicate that never
fires when its window contains no win.

V548 requires both firing and non-firing evidence for verified termination.
It permits unverified hypotheses only to choose legal discriminating actions.
It measures how each mandatory model responds as visible evidence grows. The
live shadow stays default-off and cannot update the solve registry.

## Research findings used in V548

The dated search record is in `research-references.md` under the V548 planner
marker.

| Finding | Experiment use |
|---|---|
| Hypothesis Frontier, arXiv:2608.10843 | Exp6367 keeps the strongest exactly verified factor and uses residual counterexamples to guide later live proposals. |
| Dependency-Guided Rollback Repair, arXiv:2608.10502 | Exp6369 records factor-to-decision lineage and selectively invalidates unsupported descendants. |
| ReTree, arXiv:2608.10676 | Exp6369 uses branch-local invalidation and full-reset controls while preserving independently supported state. |
| Decomposition-Induced Context-Memory Conflict, arXiv:2608.10627 | Exp6366 and Exp6371 bind parsed obligations back to raw source events and treat substitutions as harm. |
| Optimal Stopping of Self-Refining Foundation Models, arXiv:2608.10729 | Exp6367 records marginal verified gain and cost by round and uses a preregistered stop rule. |
| Active Reward Machine Inference, arXiv:2604.07480 | Exp6373 selects legal actions that distinguish competing goal automata from raw visible trajectories. |
| Zero-Shot Goal Recognition, arXiv:2605.15333 | Exp6374 measures evidence-response curves instead of one terminal goal guess. |
| Memoir, arXiv:2607.20792 | Exp6368 keeps the released factor read-only during a proposal step and commits only after exact release. |

The new constrained-Ising temperature estimator is retained as future sampler
context. The withdrawn decision-token paper is watch-only. Neither changes a
V548 prerequisite. KAN training, parser/JIT decoding, external text scoring,
hidden-state scoring, EBT pretraining, and unchanged physical-board probes stay
closed.

## Architecture

```text
                   mandatory frozen local GGUF models
             Qwen3.6-35B   Gemma-4-31B   Gemma-4-26B
                              |
                  observable child-process contract
          source hash | prompt tokens | stderr | GPU | phases
                              |
                    nonempty raw output receipt
                              |
             +----------------+----------------+
             |                                 |
    certified factor path                 live ARC path
             |                                 |
    source-bound edit parser              own frames/actions
             |                                 |
    exact verified incumbent              competing reward automata
             |                                 |
    residual counterexamples              legal disagreement action
             |                                 |
    anytime e-value release               two-sided evidence contract
             |                                 |
    versioned factor registry             reject/accept/unverifiable
             |                                 |
    dependency lineage graph              default-off action shadow
             |                                 |
    selective rollback                    no registry solve credit
             |                                 |
             +----------------+----------------+
                              |
                 independent exact safety audit
```

The exact task checker is the correctness oracle for the factor phase. The ARC
environment transition is an evaluation oracle after an action is frozen. The
live ARC agent cannot read hidden source, offline search results, or per-game
adapters. A learned factor, model score, or goal hypothesis is never an oracle.

## Phase 0: Evidence boundary and observable runtime

### Exp6363 - V547 terminal handoff and V548 queue preflight

Reconcile the seven active V547 tasks from primary artifacts and conductor
records. Preserve flags, empty-generation evidence, three repeated gate
failures, the pre-emptive skip, and the safety null. Validate all V548 IDs,
deliverables, gates, prior-failure entries, prompt endings, and model policy.

**Deliverable:** `results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json`

### Exp6364 - V548 dated source and scope freeze

Validate the V548 research marker and direct sources. Freeze three executable
lanes: observable GGUF proposals, certified factor learning with rollback, and
falsifiable live ARC goals. Record all closed and deferred scopes.

**Deliverable:** `results/experiment_6364_v548_post_marker_source_scope_freeze.json`

### Exp6365 - Three-model GGUF failure forensics and observable runtime contract

Reproduce the V547 child command in a bounded diagnostic. Retain stderr. Bind
the call to source, command, prompt, tokenizer, context capacity, dispatcher,
GPU samples, and phase timings. Test failure injection. End with one minimal
nonempty generation per mandatory model. Make no proposal-quality claim.

**Deliverable:** `results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json`

## Phase 1: Live certified learning and dependency rollback

### Exp6366 - Gated repaired three-model proposal authenticity canary

Run only when Exp6365 reports a complete observable runtime. Use fresh
executable events and all three mandatory models. Require nonempty raw output,
source-bound parsing, same-step write isolation, and exact outcome receipts.

**Deliverable:** `results/experiment_6366_repaired_live_factor_proposal_authenticity.json`

### Exp6367 - Gated verified-frontier live proposal A/B

Compare independent repeated proposals with the Hypothesis Frontier method.
Use matched calls, tokens, candidate counts, exact checks, and wall time. Keep
the strongest verified incumbent. Give later rounds only immutable residual
counterexamples. Stop on fixed budget or preregistered no-gain.

**Deliverable:** `results/experiment_6367_verified_frontier_factor_proposal_ab.json`

### Exp6368 - Gated prospective certified factor stream

Seal a chronological stream and protected future events before learning.
Compare frozen factors, V546 replay, and the live verified frontier. Keep the
active version read-only during proposal generation. The exact outcome and
e-process decide release. Record same-family and held-family retention.

**Deliverable:** `results/experiment_6368_prospective_verified_frontier_factor_learning.json`

### Exp6369 - Dependency-guided rollback stress test

Build factor-to-factor and factor-to-consumer lineage. Inject stale, poisoned,
duplicated, and misattributed releases. Compare selective rollback, full reset,
and no rollback. This task runs even if Exp6368 is blocked. It cannot promote
utility.

**Deliverable:** `results/experiment_6369_dependency_guided_factor_rollback_stress.json`

### Exp6370 - Gated default-off future consumer A/B

Run only when prospective learning and rollback are ready. Freeze all writes.
Compare frozen, replay, and certified-live registries on untouched future
events. Report exact yield, false accepts, abstention, harmful interventions,
verification cost, latency, and retention. Keep the consumer default-off.

**Deliverable:** `results/experiment_6370_default_off_certified_factor_consumer_ab.json`

### Exp6371 - Independent live-learning and consumer audit

Attack process substitution, source-artifact drift, empty raw output, context
overflow, source-to-obligation substitution, family identity shortcuts,
same-step writes, evidence duplication, optional-stopping reset, protected
event leakage, dependency corruption, incomplete rollback, and unsafe consumer
promotion. The audit runs regardless of utility gates.

**Deliverable:** `results/experiment_6371_live_learning_consumer_and_rollback_audit.json`

## Phase 2: Falsifiable live ARC goals

### Exp6372 - Two-sided ARC goal evidence contract

Implement accepted, rejected, and unverifiable states. Require sensitivity and
specificity evidence for verified termination. An unverified goal may guide a
probe, but it cannot terminate a level. Keep the contract default-off.

**Deliverable:** `results/experiment_6372_arc_two_sided_goal_evidence_contract.json`

### Exp6373 - Gated live-path active reward-machine discriminator

Represent a bounded set of goal explanations as small automata over visible
events. Choose a legal action only when predicted observations distinguish
hypotheses. Prove reachability from `E3AgentPolicy` and `make_carnot_agent`.
Use no hidden source, offline ground-truth search, or per-game adapter.

**Deliverable:** `results/experiment_6373_arc_active_reward_machine_discriminator.json`

### Exp6374 - Gated three-model ARC evidence-response calibration

Use matched trajectory prefixes and all three mandatory GGUF families. Compare
the current gate, frozen-prior control, passive accumulation, and active
two-sided evidence. Measure false accepts, admission precision, abstention,
calibration, and response to added evidence. Freeze predictions before the
evaluation oracle is read. Make no solve claim.

**Deliverable:** `results/experiment_6374_arc_goal_evidence_response_calibration.json`

### Exp6375 - Gated default-off live ARC active-goal shadow

Run only after calibration improves precision without increasing false accepts.
Use the normal live agent entrypoint and the middle mandatory MoE model. Compare
route off and active-goal shadow at matched budgets. Measure hypothesis
elimination, legal action changes, transition evidence, deadlines, and cost.
Do not update the solve registry.

**Deliverable:** `results/experiment_6375_arc_default_off_active_goal_shadow.json`

## Phase 3: Adversarial reconciliation

### Exp6376 - V548 adversarial capstone

Aggregate every present task through the artifact summarizer. Recompute gates
and adversarial status. Separate clean, null, blocked, missing, flagged, and
retired evidence. Decide the runtime, FR-11, rollback, consumer, ARC, and
hardware states without rerunning an upstream experiment.

**Deliverable:** `results/experiment_6376_v548_adversarial_capstone.json`

## Dependency graph

```text
Exp6363 -> Exp6364 -> Exp6365 -> Exp6366 -> Exp6367 -> Exp6368
                                      |                    |
                                      |                    +-> Exp6369
                                      |                           |
                                      |                    Exp6370 -> Exp6371
                                      |
                                      +-> Exp6372 -> Exp6373 -> Exp6374 -> Exp6375
                                                                  |
                                    all present, blocked, or missing evidence
                                                                  |
                                                               Exp6376
```

Exp6366 is gated on runtime observability. Exp6367 is gated on proposal
authenticity. Exp6368 is gated on a positive verified-frontier delta. Exp6370
is gated on both prospective learning and rollback readiness. Exp6373 is gated
on the two-sided contract. Exp6374 is gated on both ARC substrates. Exp6375 is
gated on positive admission-precision change, no false-accept increase, and
calibration readiness. Exp6369, Exp6371, and Exp6376 run even if utility tasks
are blocked.

No V548 task requires Exp6353, Exp6354, or Exp6355. Those upstream identities
belong to a retired V547 gate chain. The new chain starts from the materially
changed Exp6365 runtime contract. Every matching task records prior failures
and retires itself if the same verdict recurs.

## Hardware requirements

| Resource | Use | Rule |
|---|---|---|
| Two local RTX 3090 GPUs | Sequential model loading for Exp6365-Exp6370 and Exp6374; matched live ARC model use in Exp6375 | Record model placement, offload support, task-linked GPU samples, peak memory, unload, and cleanup. Block a measured cell if its required model cannot run. |
| Host CPU and RAM | Embedded GGUF tokenization, exact validators, e-process, registry, dependency graph, ARC contracts, and audits | Record exact-check wall time, errors, and peak host memory where relevant. |
| Local disk | Raw stdout/stderr, source snapshots, prompts, manifests, factor versions, lineage, and rollback snapshots | Check free space first. Hash every protected file and sidecar. |
| Network | Source validation in Exp6364 and normal ARC SDK traffic in Exp6375 | Block only the affected cell when unavailable. Never invent a receipt. |
| GateMate, KV260, PolarFire, NPU, Extropic TSU, or Kona | Not used | There is no changed authenticated physical state or runnable third-party interface. Run no probe and make no speed, energy, power, or availability claim. |

Every LLM experiment must use `cached_sota_pair()` and include at least one of
the three mandatory hub IDs. The main factor experiments use all three:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6375 uses `unsloth/gemma-4-26B-A4B-it-GGUF` as its headline live ARC model.
The legacy Qwen3.5-0.8B and Gemma-4-E4B models may appear only in CPU smoke
tests. They cannot populate a measured or headline cell. Use only the embedded
GGUF tokenizer. Never use `AutoTokenizer` for a GGUF artifact.

## Milestone success conditions

V548 succeeds scientifically if it gives honest terminal answers to three
questions:

1. Can Carnot produce observable, source-bound, nonempty live outputs from all
   three mandatory local GGUF families?
2. Does a verified-incumbent proposal frontier improve protected future exact
   outcomes, and can dependency-guided rollback remove harmful descendants
   without erasing independently supported factors?
3. Does a two-sided active-evidence contract reduce vacuous ARC goal admission
   on the live path without hidden information, false-accept growth, or solve
   laundering?

A null or blocked result is terminal evidence. It must not become readiness.
The milestone makes no hardware speed claim, no KAN claim, no parser/JIT claim,
no external-text or hidden-state verifier claim, and no ARC solve claim.

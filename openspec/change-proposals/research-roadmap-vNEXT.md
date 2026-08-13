# Research Roadmap V549: Canonical Factor Transport and Prospective Learning

**Milestone:** `2026.08.549`
**Date:** 2026-08-13
**Status:** Proposed
**Task range:** Exp6377-Exp6390
**Execution file:** `research-roadmap-next.yaml`

## Milestone thesis

V548 closed the local runtime question. All three mandated GGUF families loaded
through llama.cpp, used their embedded tokenizers, offloaded to the two RTX 3090
GPUs, and returned nonempty completions. It did not close the factor-proposal
question. Every Exp6366 completion failed parsing, so the exact checker received
zero calls.

The failure has three different observed forms. Qwen3.6 used its full
192-token allowance for reasoning. Gemma-4-31B repeated one token. Gemma-4-26B
started the target JSON and stopped inside `edit_source_spans`. The V548 prompt
also repeated the target object inside a larger payload. V549 treats this as a
transport-contract problem before it treats it as a reasoning problem.

V549 creates one canonical factor-edit schema. That source generates the
prompt, example, validator, and per-model output-token lower bound. A bounded
three-arm canary separates instruction drift, capacity, and repetition. It uses
no grammar decoding, parser retry, post-hoc repair, or token-budget increase as
a stand-alone mechanism. The canary must produce source-bound, parse-valid
objects from every mandated family before factor learning runs.

The learning phase then tests a verified-incumbent proposal frontier on fresh
events. A chronological continuous self-learning trial may commit only an
exactly released factor. A dependency graph can revoke unsupported descendants.
A future consumer stays default-off until both learning and rollback pass.

The ARC phase executes the goal-evidence work that never entered the four-task
V548 active queue. It adds a two-sided goal contract, an active reward-machine
discriminator, a three-model evidence-response calibration, and a default-off
live shadow. It makes no game or level solve claim.

## What V548 proved

| Evidence | Result | V549 consequence |
|---|---|---|
| Exp6363 terminal handoff | `blocked_v548_queue_incomplete`; the active queue had four tasks, not the 14 proposal tasks | Reconcile only Exp6363-Exp6366. Validate the complete V549 queue before activation. |
| Exp6364 source freeze | `complete_no_scope_change`; no strict post-marker finding changed the V548 scope | Start a new dated marker. Keep the same scientific lanes unless a direct source changes them. |
| Exp6365 GGUF runtime | `gguf_runtime_observability_ready_score=1.0`; all three models loaded, offloaded, generated, and cleaned up with complete process receipts | Reuse this runtime. Do not spend another milestone on child-process diagnosis. |
| Exp6366 raw generation | Three nonempty outputs, 1,991 total bytes, zero timeouts, and authenticated GPU execution | The model path is live. The next gate is structured transport, not process authenticity. |
| Exp6366 parsing | Zero valid objects and zero exact-checker calls across all three families | Build one canonical schema surface and measure capacity before retrying authenticity. |
| Exp6366 Qwen row | The output used the completion allowance for reasoning and never emitted JSON | Test an explicit output-only protocol and record thinking leakage as a transport failure. |
| Exp6366 dense Gemma row | The output collapsed into repeated `own` tokens | Add a preregistered repetition control and preserve the raw failure class. |
| Exp6366 MoE Gemma row | The output began valid JSON and ended mid-field | Compute the output lower bound with each embedded tokenizer. Keep context and completion margins separate. |
| Proposal-only V548 tasks | Exp6367-Exp6376 were never in the active roadmap and have no execution evidence | Use new experiment IDs. Do not cite proposal text as results. |

## The three largest gaps

### Gap 1: Live factor edits cannot cross the structured boundary

Carnot can now run every required local model, but no model output reached the
exact checker. The current prompt has more than one instruction source and an
output allowance that was not derived from the serialized target. This blocks
every later learning claim.

V549 adds a canonical schema hash, prompt-schema drift checks, tokenizer-based
capacity math, bounded repetition controls, and separate transport and semantic
scores. Parse validity is necessary. It is never sufficient for release.

### Gap 2: FR-11 still has no prospective live-model benefit

V546 proved replay release, retention, rollback, and bounded growth. V548 did
not reach a factor-learning experiment. Carnot has not shown that a live local
model improves untouched future exact outcomes in a chronological stream.

V549 compares independent proposals with a verified-incumbent frontier. It
then runs continuous self-learning under a read-only-then-commit transaction.
The exact checker and anytime evidence process control release. A dependency
graph supports selective rollback. The consumer stays disabled unless both
paths qualify.

### Gap 3: ARC goal admission is still one-sided

Exp6258 accepted 21 false goals and five true goals among 29 measurable
predicates. Its acceptance precision was 0.1923. The proposed V548 repair did
not execute.

V549 requires both firing and non-firing evidence before a goal can terminate a
level. Unverified hypotheses may choose legal probes only. Active reward
machines select probes that distinguish hypotheses from the live agent's own
frames and actions. The experiments update no solve registry.

## Research findings used in V549

The dated search record is in `research-references.md` under the V549 planner
marker.

| Finding | Experiment use |
|---|---|
| Structured-output schema descriptions, arXiv:2608.08254 | Exp6379 derives every instruction surface from one canonical schema. Exp6380 tests a bounded evidence field and prevents prompt-schema drift. |
| Structured-output control for software engineering, arXiv:2606.09395 | Exp6380 reports syntax, structure, source binding, and exact semantics separately. Syntax success cannot become a learning claim. |
| SchemaBench, arXiv:2502.18878 | Exp6379 reuses the failure taxonomy and validator discipline. It does not fine-tune the GGUF models. |
| Hypothesis Frontier, arXiv:2608.10843 | Exp6381 keeps the strongest exactly verified factor and presents only residual immutable failures in later rounds. |
| Dependency-Guided Rollback, arXiv:2608.10502, and ReTree, arXiv:2608.10676 | Exp6383 compares selective descendant invalidation with full reset and no rollback. |
| Memoir, arXiv:2607.20792 | Exp6382 keeps the active factor version read-only during proposal generation and commits after exact release only. |
| Active Reward Machine Inference, arXiv:2604.07480 | Exp6387 chooses legal actions that distinguish bounded goal automata. |
| Zero-Shot Goal Recognition, arXiv:2605.15333 | Exp6388 measures evidence-response curves across the three mandated model families. |

The Extropic Z1 tapeout is product-roadmap evidence. First-party material places
early access in 2027. Carnot has no authenticated device or simulator API. Kona
still exposes no local weights or reproducible runtime. KAN training,
parser/JIT decoding, external text scoring, hidden-state scoring, EBT
pretraining, and unchanged physical-board probes stay closed.

## Architecture

```text
                 mandatory frozen local GGUF families
          Qwen3.6-35B   Gemma-4-31B   Gemma-4-26B
                              |
                  observable V548 llama.cpp runtime
                              |
          canonical factor-edit instruction surface
      schema hash | prompt | example | validator | capacity
                              |
          three-arm source-bound transport canary
                              |
                +-------------+-------------+
                |                           |
       certified factor path          live ARC path
                |                           |
       verified incumbent              own frames/actions
                |                           |
     residual counterexamples      two-sided goal evidence
                |                           |
      chronological CSL stream      reward-machine frontier
                |                           |
     exact release and commit        legal disagreement probe
                |                           |
    factor dependency lineage     three-model calibration
                |                           |
       selective rollback          default-off live shadow
                |                           |
   default-off future consumer     no solve-registry update
                |                           |
                +-------------+-------------+
                              |
                 independent safety audit
                              |
                    adversarial capstone
```

The exact task checker is the correctness oracle for factor learning. The ARC
environment transition is an evaluation oracle after the agent freezes an
action. Model text, parsed JSON, learned scores, and reward hypotheses are not
oracles.

## Phase 0: Evidence boundary and factor transport

### Exp6377 - V548 terminal handoff and V549 queue preflight

Reconcile the four active V548 tasks from their terminal artifacts and
conductor records. Preserve the Exp6363 blocked state and Exp6366 null. Validate
all 14 V549 task IDs, deliverables, gates, prior failures, model policy, prompt
contracts, and exclusion-manifest boundaries before activation.

**Deliverable:** `results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json`

### Exp6378 - V549 dated source and scope freeze

Validate the V549 research marker and direct sources. Freeze three executable
lanes: canonical factor transport, prospective certified self-learning with
rollback, and falsifiable live ARC goals. Record Extropic, Kona, KAN, and local
hardware as non-executable unless a dated receipt changes.

**Deliverable:** `results/experiment_6378_v549_post_marker_source_scope_freeze.json`

### Exp6379 - Canonical factor-edit transport contract

Replay the three Exp6366 raw failures without model calls. Create one canonical
schema object that generates the prompt fragment, compact example, validator,
and per-tokenizer output lower bound. Add drift, truncation, thinking leakage,
repetition, and semantic-boundary tests. Make no claim about model quality.

**Deliverable:** `results/experiment_6379_canonical_factor_edit_transport_contract.json`

### Exp6380 - Gated three-model canonical transport canary

Compare the frozen V548 prompt, the canonical prompt at the old allowance, and
the canonical prompt with computed headroom and a fixed repetition control.
Use all three mandated models. Require at least one source-bound parse-valid
object and an exact-checker call from each family. Keep semantic results
separate from transport readiness.

**Deliverable:** `results/experiment_6380_three_family_canonical_factor_transport_canary.json`

## Phase 1: Prospective certified self-learning

### Exp6381 - Gated verified-frontier live proposal A/B

Compare independent repeated proposals with a verified-incumbent frontier at
matched calls, output capacity, exact checks, and time. Later rounds see only
immutable residual counterexamples. Stop at a fixed budget or a preregistered
no-gain boundary.

**Deliverable:** `results/experiment_6381_verified_frontier_live_factor_proposal_ab.json`

### Exp6382 - Gated chronological continuous self-learning

Seal past, release, and future partitions before learning. Compare frozen,
V546 replay, and live-frontier factors. Keep the active registry read-only
during proposal generation. Commit only after exact release. Measure forward
transfer, retention, negative transfer, cost, quarantine, restart, and rollback.

**Deliverable:** `results/experiment_6382_chronological_verified_factor_self_learning.json`

### Exp6383 - Dependency-guided rollback stress test

Build factor-to-factor and factor-to-decision lineage. Inject stale, poisoned,
duplicated, misattributed, and partially supported releases. Compare selective
rollback, full reset, and no rollback. This task runs even when learning is
blocked and cannot promote utility.

**Deliverable:** `results/experiment_6383_dependency_guided_factor_rollback_stress.json`

### Exp6384 - Gated default-off future consumer A/B

Run only after prospective learning and rollback both qualify. Freeze writes.
Compare frozen, replay, and certified-live registries on untouched future
events. Report exact yield, false accepts, abstention, harm, latency,
verification cost, and retention. Keep the consumer default-off.

**Deliverable:** `results/experiment_6384_default_off_certified_factor_consumer_ab.json`

### Exp6385 - Independent learning, rollback, and consumer audit

Attack process substitution, schema drift, capacity undercount, repeated-token
acceptance, source substitution, same-step writes, evidence duplication,
optional-stopping reset, held-event leakage, lineage corruption, incomplete
rollback, and unsafe consumer promotion. Run even if upstream gates block.

**Deliverable:** `results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json`

## Phase 2: Falsifiable live ARC goals

### Exp6386 - Two-sided ARC goal-evidence contract

Implement accepted, rejected, and unverifiable states. Verified termination
requires both sensitivity and specificity evidence. An unverified hypothesis
may guide a probe but cannot terminate a level. Make no solve claim.

**Deliverable:** `results/experiment_6386_arc_two_sided_goal_evidence_contract.json`

### Exp6387 - Gated live-path active reward-machine discriminator

Represent a bounded set of goal hypotheses as small automata over visible
events. Choose an action only when legal candidate outcomes distinguish
hypotheses. Prove reachability from `E3AgentPolicy` and `make_carnot_agent`.
Use no hidden source, offline ground-truth search, or per-game adapter.

**Deliverable:** `results/experiment_6387_arc_active_reward_machine_discriminator.json`

### Exp6388 - Gated three-model ARC evidence-response calibration

Use matched trajectory prefixes and all three mandated model families. Compare
the current gate, a frozen-prior control, passive evidence accumulation, and
active two-sided evidence. Freeze predictions before reading the evaluation
oracle. Report false accepts, precision, abstention, calibration, and response
to added evidence. Make no solve claim.

**Deliverable:** `results/experiment_6388_arc_goal_evidence_response_calibration.json`

### Exp6389 - Gated default-off live ARC active-goal shadow

Run only if calibration improves admission precision without increasing false
accepts. Use the normal live agent entrypoint and the middle mandated MoE model.
Compare route-off and active-goal shadow at matched budgets. Do not update the
solve registry.

**Deliverable:** `results/experiment_6389_arc_default_off_active_goal_shadow.json`

## Phase 3: Adversarial reconciliation

### Exp6390 - V549 adversarial capstone

Aggregate every present task through the artifact summarizer. Recompute every
gate and adversarial status. Separate clean, null, blocked, missing, flagged,
and retired evidence. Decide transport, FR-11, rollback, consumer, ARC, and
hardware states without rerunning an upstream experiment.

**Deliverable:** `results/experiment_6390_v549_adversarial_capstone.json`

## Dependency graph

```text
Exp6377 -> Exp6378 -> Exp6379 -> Exp6380 -> Exp6381 -> Exp6382
                  |                                      |
                  +---------------> Exp6383 -------------+-> Exp6384

Exp6379 -> Exp6385 <- Exp6381, Exp6382, Exp6383, and Exp6384

Exp6378 -> Exp6386 -> Exp6387 -> Exp6388 -> Exp6389

all present, blocked, missing, flagged, or retired evidence -> Exp6390
```

Exp6380 is gated on the deterministic transport contract. Exp6381 is gated on
three-family transport authenticity. Exp6382 is gated on frontier readiness and
a positive future exact-yield delta. Exp6384 is gated on both prospective
learning and rollback readiness. Exp6387 is gated on the two-sided ARC contract.
Exp6388 is gated on live reward-machine reachability. Exp6389 is gated on
calibration readiness, positive precision delta, and no false-accept increase.
Exp6383, Exp6385, and Exp6390 run even when utility tasks are blocked.

The new factor chain declares Exp6366, Exp6353, Exp6354, Exp6355, and Exp6356
where task scope matches. The ARC contract declares the Exp6258 false-accept
result. Every entry contains the prior verdict, a changed mechanism, and
`retire_if_same_verdict: true`. No task requires a retired upstream ID.

## Hardware requirements

| Resource | Use | Rule |
|---|---|---|
| Two local RTX 3090 GPUs | Sequential GGUF loading for Exp6380-Exp6382, Exp6384, Exp6388, and Exp6389 | Record model placement, offload, task-linked GPU samples, peak memory, unload, and cleanup. Block only the affected measured cell if a required model cannot run. |
| Host CPU and RAM | Canonical schema generation, embedded tokenization, exact validators, evidence process, lineage graph, ARC contracts, and audits | Record exact-check cost, errors, and peak host memory where relevant. |
| Local disk | Raw outputs, prompts, canonical schema, event manifests, factor versions, lineage, rollback snapshots, and ARC traces | Check free space first. Hash every protected file and sidecar. |
| Network | Source validation in Exp6378 and normal ARC SDK traffic in Exp6389 | Record unavailable cells. Never invent a receipt. |
| GateMate, KV260, PolarFire, NPU, Extropic TSU, or Kona | Not used | No authenticated physical or runnable third-party state changed. Make no speed, energy, power, latency, or availability claim. |

Every experiment that invokes an LLM must use `cached_sota_pair()` and include
at least one of these IDs in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The factor canary, frontier, continuous learner, consumer, and ARC calibration
use all three. Exp6389 uses `unsloth/gemma-4-26B-A4B-it-GGUF` as its headline
model. Legacy small models may appear only in CPU smoke tests. They cannot
populate a measured or headline cell. Use the embedded GGUF tokenizer only.
Never use `AutoTokenizer` for a GGUF artifact.

## Decentralization

V549 keeps models, exact validators, learning state, and evaluation artifacts on the local user-controlled host.

## Milestone success conditions

V549 succeeds scientifically if it gives honest terminal answers to four
questions:

1. Can one canonical schema and measured capacity contract produce
   source-bound, parse-valid factor edits from all three mandated local models?
2. Does a verified-incumbent frontier improve untouched future exact outcomes,
   and can chronological continuous self-learning retain that gain without
   leakage or same-step writes?
3. Can dependency-guided rollback remove harmful descendants while preserving
   independently supported factors, and can a default-off consumer use only a
   qualified registry?
4. Does two-sided active goal evidence reduce ARC false admission on the live
   path without hidden information, false-accept growth, or solve laundering?

A null or blocked result is terminal evidence. It must not become readiness.
The milestone makes no grammar-decoding claim, parser-repair claim, hidden-state
or external-text verifier claim, KAN claim, hardware-speed claim, or ARC solve
claim.

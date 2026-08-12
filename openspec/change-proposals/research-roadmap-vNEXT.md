# Research Roadmap V547: Prospective Factor Learning and Falsifiable Goals

**Milestone:** `2026.08.547`
**Date:** 2026-08-12
**Status:** Proposed
**Task range:** Exp6350-Exp6362
**Execution file:** `research-roadmap-next.yaml`

## Milestone thesis

V546 proved that Carnot can certify factor updates in a bounded replay. It did
not prove that a local model can make the proposals in a real generation loop.
It also did not prove that released factors help a later consumer.

V547 turns the replay result into a prospective, local-GGUF experiment. The
model reads one approved factor version. It proposes an edit from an exact
counterexample. The exact checker labels the outcome. The e-process decides
whether to release. The next event can read only a released version. Base model
weights stay frozen.

The ARC phase fixes a separate verifier defect. The current live goal gate is
mostly a specificity test. A predicate that is always false can pass when no
win appears in the attempt window. V547 adds a two-sided evidence contract and
an explicit `unverifiable` state. It then tests active reward-machine
hypotheses on the live agent's own trajectory path. It makes no solve claim.

## What V546 proved

| Evidence | Result | V547 consequence |
|---|---|---|
| Exp6337 terminal handoff | Artifact exists, but the conductor flagged short duration and missing method evidence | Preserve the flag. Use a bounded transition task. Do not promote the artifact as clean evidence. |
| Exp6338 source freeze | `complete_null`; no strict post-marker source changed scope | Start a new dated window after the V547 marker. |
| Exp6339 prefix substrate | Exact bounded prefix checks passed | Keep the substrate as a utility-null component. Do not extend it. |
| Exp6340 parser/JIT canary | No semantic-diversity gain at matched cost | Close parser/JIT constrained generation. Exp6341 stayed gate-blocked. |
| Exp6342 e-value ledger | Anytime release logic passed bounded synthetic checks | Reuse it as the release authority in a real chronological stream. |
| Exp6343 factor lifecycle | Evidence, retention, rollback, and bounded growth passed exact checks | Reuse the registry. Keep proposal and commit separated. |
| Exp6344 proposal calibration | Counterexample-directed rows beat repeated sampling in deterministic replay | Repeat only with real autoregressive calls and raw-output receipts. |
| Exp6345 certified evolution | Positive on six replay events with tokenizer-only model access | Replace replay outcomes with fresh executable events and real proposals. |
| Exp6346 safety audit | Synthetic attack replay failed closed | Attack the real generation, parsing, release, restart, and consumer path. |
| Exp6347 and Exp6348 ARC influence | Route reached the live path and changed action order without solve credit | Fix the goal-admission defect before asking for more route influence. |
| Exp6349 capstone | Certified learning closed only inside exact replay boundaries; ARC closed with no solve | Keep both claim boundaries explicit. |

## The three largest gaps

### Gap 1: Continuous learning has no real proposal evidence

The positive V546 learning result used deterministic proposal replay. The
artifacts loaded local tokenizers, but they did not run live autoregressive
generation for factor edits. Model-family comparisons therefore do not yet
measure model behavior.

V547 requires raw generation receipts from Qwen3.6-35B-A3B, Gemma-4-31B, and
Gemma-4-26B-A4B. It uses fresh executable events. It records parse failures,
invalid edits, exact outcomes, and the full release chronology.

### Gap 2: Certified factors have no prospective consumer value

The factor registry, rollback, and e-process are mechanisms. They have not
improved a later local-model consumer on events that were unavailable when the
factor was proposed. A safe memory with no downstream value does not close
FR-11.

V547 freezes future same-family and held-family events before learning. A
default-off consumer compares frozen memory, replay learning, and certified
live learning at matched generation and exact-check budgets. Exact validators
retain veto authority.

### Gap 3: ARC goal admission is one-sided and over-admits vacuous goals

Exp6258 found 21 false accepts, 5 true accepts, 3 true rejects, and no false
rejects among 29 measurable predicates. Acceptance precision was 0.1923. The
problem is not a loose threshold. The gate lacks sensitivity evidence when a
level-1 attempt contains no win.

V547 separates an exploratory goal hypothesis from a verified termination
predicate. A goal without positive evidence is `unverifiable`, not accepted.
Active reward-machine hypotheses select legal actions that distinguish
competing explanations. The work stays default-off and on the live agent path.

## Research findings used in V547

The full search record is in `research-references.md` under the V547 planner
marker.

| Finding | Experiment use |
|---|---|
| Active Reward Machine Inference From Raw State Trajectories, arXiv:2604.07480 | Exp6357-Exp6360 infer competing goal automata from raw visible transitions and choose discriminating legal actions. |
| Zero-Shot Goal Recognition with Large Language Models, arXiv:2605.15333 | Exp6359 measures how goal quality changes as evidence prefixes grow. |
| Memoir, arXiv:2607.20792 | Exp6352-Exp6355 keep the active factor version read-only during proposal generation and commit only after release. |
| Solver-Hard Is Not Model-Hard, arXiv:2607.17047 | Exp6353 and Exp6354 balance executable structure and surface form instead of using solver effort as model difficulty. |
| Distributional EBMs for Structured LLM Reasoning, arXiv:2605.18871 | Exp6355 uses exact penalties, family-blind controls, and abstention. Learned scores cannot approve output. |
| The Verification Horizon, arXiv:2606.26300 | Exp6356 attacks reward saturation, replay leakage, and a changing proposal distribution. |

## Architecture

```text
                         frozen local GGUF generators
                  Qwen3.6-35B   Gemma-4-31B   Gemma-4-26B
                              |           |
                              +-----+-----+
                                    |
                           raw proposal or goal text
                                    |
                    +---------------+----------------+
                    |                                |
          certified factor path               live ARC goal path
                    |                                |
       read released factor version       own frames and actions only
                    |                                |
       minimized exact counterexample      competing reward automata
                    |                                |
       bounded factor-edit proposal        legal disagreement probe
                    |                                |
       schema parse and exact outcome      more visible transitions
                    |                                |
       immutable e-process release         two-sided evidence contract
                    |                                |
       commit, quarantine, or rollback     accept / reject / unverifiable
                    |                                |
       default-off future consumer         default-off action shadow
                    |                                |
                    +---------------+----------------+
                                    |
                     exact audit and protected replay
```

The exact task checker is the correctness oracle for factor experiments. The
observable ARC level counter is evaluation evidence. A learned factor or goal
hypothesis is never an oracle. It may propose or rank. It may not approve its
own output.

## Phase 0: Evidence boundary and source freeze

### Exp6350 - Bounded V546 terminal handoff

Reconcile Exp6337-Exp6349. Preserve the Exp6337 conductor flag. Record that the
prefix utility task was gate-blocked. Separate real inference, tokenizer-only
access, deterministic replay, and aggregation substrates.

**Deliverable:** `results/experiment_6350_v547_bounded_terminal_handoff.json`

### Exp6351 - V547 source and scope freeze

Validate the new planner marker and direct sources. Freeze the two scientific
lanes: prospective factor learning and falsifiable ARC goal discovery. Freeze
the no-hardware rule and the closed parser/JIT lane.

**Deliverable:** `results/experiment_6351_v547_post_marker_source_scope_freeze.json`

## Phase 1: Real-generation certified continuous learning

### Exp6352 - Local-GGUF factor proposal authenticity preflight

Use all three mandatory local GGUF models through `cached_sota_pair()`. Run
small, real autoregressive calls against fresh executable failure events. Prove
that raw outputs come from the model process, parse through the bounded edit
schema, and do not read an unapproved write.

**Deliverable:** `results/experiment_6352_live_factor_proposal_authenticity_preflight.json`

### Exp6353 - Real counterexample-directed proposal A/B

Run only after the authenticity preflight passes. Compare repeated sampling,
stability-only proposals, and minimized-counterexample proposals. Use matched
calls, tokens, candidates, exact checks, and time. Balance formula structure
and surface form. The endpoint is future exact success per matched cost.

**Deliverable:** `results/experiment_6353_live_counterexample_factor_proposal_ab.json`

### Exp6354 - Prospective read-only-then-commit learning stream

Run only if live counterexample proposals beat the matched control. Seal a
chronological stream and protected future events. Compare frozen factors,
the V546 replay rule, and real certified evolution. A proposal reads only the
last released version. Exact outcomes and the e-process control every commit.

**Deliverable:** `results/experiment_6354_prospective_live_certified_factor_learning.json`

### Exp6355 - Default-off future consumer A/B

Run only if Exp6354 is ready. Apply the released registry to fresh future
events without further writes. Compare exact yield, abstention, false accepts,
checker cost, and latency across same-family and held-family cells. Keep the
consumer default-off.

**Deliverable:** `results/experiment_6355_default_off_certified_factor_consumer_ab.json`

### Exp6356 - Independent live-learning safety audit

Attack raw-output substitution, deterministic-replay laundering, same-step
read/write coupling, duplicated evidence, optional-stopping reset, model-family
identity shortcuts, parser ambiguity, protected-event leaks, unsafe release,
restart corruption, registry growth, and rollback. Safety cannot promote a
null utility result.

**Deliverable:** `results/experiment_6356_live_certified_learning_safety_audit.json`

## Phase 2: Falsifiable live ARC goal discovery

### Exp6357 - Two-sided goal evidence contract

Implement accepted, rejected, and unverifiable states. Require sensitivity and
specificity evidence for verified termination. Permit an unverified goal only
as a probe hypothesis. Keep the new contract default-off.

**Deliverable:** `results/experiment_6357_arc_two_sided_goal_evidence_contract.json`

### Exp6358 - Active reward-machine discriminator

Represent competing goal explanations as small reward automata over visible
events. Select a legal next action only when predicted observations distinguish
hypotheses. Use no hidden game source, offline ground-truth search, or per-game
adapter. Prove that the operator is reachable from the live agent path.

**Deliverable:** `results/experiment_6358_arc_active_reward_machine_discriminator.json`

### Exp6359 - Three-model evidence-response calibration

Run all three mandatory local GGUF models on matched, target-licensed
trajectory prefixes. Compare the current specificity-only gate, a frozen-prior
control, and two-sided active evidence. Measure false accepts, admission
precision, abstention, and response to added evidence. Use development-proxy
win evidence only after each prediction is frozen. Make no solve claim.

**Deliverable:** `results/experiment_6359_arc_goal_evidence_response_calibration.json`

### Exp6360 - Default-off live ARC active-goal shadow

Run only if calibration improves admission precision without a false-accept
increase. Use the normal live agent entrypoint and fresh own-attempt windows.
Compare route off and active-goal shadow at matched budgets. Measure hypothesis
elimination, legal action changes, exact observed transition quality, and cost.
Do not update the solve registry.

**Deliverable:** `results/experiment_6360_arc_default_off_active_goal_shadow.json`

### Exp6361 - ARC provenance and oracle audit

Attack hidden-source access, source-code reads, offline BFS, hand adapters,
per-game calibration, evaluator leakage, synthetic-window substitution,
off-path solvers, duplicate solve credit, and registry mutation. Recompute the
live-path and no-solve receipts.

**Deliverable:** `results/experiment_6361_arc_active_goal_provenance_audit.json`

## Phase 3: Reconciliation

### Exp6362 - V547 adversarial capstone

Recompute all dependencies and gates. Separate clean, null, blocked, flagged,
and retired outcomes. Confirm real generation receipts, factor chronology,
oracle boundaries, ARC provenance, tests, spec coverage, hardware non-use, and
operations reconciliation.

**Deliverable:** `results/experiment_6362_v547_adversarial_capstone.json`

## Dependency graph

```text
Exp6350 -> Exp6351
               |
               +-> Exp6352 -> Exp6353 -> Exp6354 -> Exp6355 -> Exp6356
               |
               +-> Exp6357 -> Exp6358 -> Exp6359 -> Exp6360 -> Exp6361
                                      \----------------------/
                                               |
                                           Exp6362
```

Exp6353 is gated on the Exp6352 authenticity score. Exp6354 is gated on a
positive live proposal delta. Exp6355 is gated on certified live-learning
readiness. Exp6358 is gated on the two-sided contract. Exp6359 is gated on both
ARC substrates. Exp6360 is gated on calibrated admission precision and zero
false-accept increase. Audits run even when utility tasks are blocked.

## Hardware requirements

| Resource | Use | Rule |
|---|---|---|
| Two local RTX 3090 GPUs | Sequential loading and inference for the three mandatory GGUF models | Record placement, offload, peak memory, unload, and per-model file hashes. Block measured cells if a required model is unavailable. |
| Host CPU and RAM | Exact validators, e-process, lifecycle, artifact replay, and ARC audits | Record exact-check wall time and errors. |
| Local disk | Raw generations, sealed manifests, registry versions, and rollback snapshots | Check capacity before generation. Hash every protected sidecar. |
| Network | Source validation in Exp6351 and normal ARC SDK access when used in Exp6360 | Block the affected task if the required endpoint is unavailable. Never invent a receipt. |
| GateMate, KV260, PolarFire, NPU, or TSU | Not used | V546 supplied no new authenticated physical state. Run no unchanged hardware probe. |

The legacy Qwen3.5-0.8B and Gemma-4-E4B models may appear only in CPU unit
tests. They cannot populate a measured or headline cell.

## Milestone success conditions

V547 succeeds scientifically if it gives an honest answer to both questions:

1. Can real local-model factor proposals, released by exact evidence, improve a
   protected future consumer at matched cost without unsafe commits?
2. Can a two-sided, active-evidence goal contract reduce vacuous ARC goal
   admissions on the live agent path without hidden information or solve
   laundering?

A null or blocked answer is terminal evidence. It must not be rewritten as
readiness. Parser/JIT constrained generation remains closed. Hardware remains
out of scope until a new physical receipt exists.

# Research Roadmap vNEXT: Constraint-First Proofs and Safe Online Memory

**Milestone:** 2026.08.573  
**Planning date:** 2026-08-24  
**Experiment range:** Exp6585-Exp6596  
**Execution file:** research-roadmap-next.yaml  
**Status:** planned

## Purpose

Milestone 572 closed with six terminal tasks. It proved that the source and
joint-method protocol is executable. It produced valid four-row source shards
for the dense and MoE Gemma families. The Qwen task blocked before model load
because the repo-wide suite was used as a launch precondition. That precondition
was later corrected. The final three-family audit then exceeded its hard limit
three times and wrote no artifact.

Milestone 573 does not repeat the unfinished three-family aggregate. It tests a
new research question from arXiv:2608.05254: can a byte-frozen constraint-first
stage improve exact-valid answers from current local models? It uses one model
per task, preserves both generation stages, and gives release authority to the
existing exact checker. It then tests reversible online constraint memory on
the audited rows. A final side task refines the ARC trajectory supervisor from
live receipts.

## What milestone 572 proved

| Item | Terminal evidence | Meaning for 573 |
|---|---|---|
| Exp6579 recovery contract | Complete terminal recovery and one-family execution contract | Keep model work decomposed. Preserve periodic row checkpoints and terminal no-artifact receipts. |
| Exp6580 source and joint-method protocol | Source-method and joint-method contracts closed | Reuse the exact source binding, fixture, authority, and failure rules. Add CFR before new inference. |
| Exp6581 Qwen source shard | Blocked before model load because `verification_commands` timed out | This was an environmental precondition failure, not a Qwen quality result. The repo-wide suite no longer belongs in model launch checks. |
| Exp6582 Gemma 4 31B source shard | Complete, four immutable rows and four checkpoints | The dense flagship local path is usable. Four rows are evidence of runtime, not decision-grade science. |
| Exp6583 Gemma 4 26B-A4B source shard | Complete, four immutable rows and four checkpoints | The middle MoE local path is usable. It remains available as a smoke and replication family, but it is not required for this bounded milestone. |
| Exp6584 three-family audit | Three hard limits near 4,800 seconds; no artifact | Retire monolithic all-family audits. Independent reducers must consume completed, bounded artifacts and perform no model load. |
| Repo-wide test state | The isolated suite is known to be red and can rewrite tracked results | Measure it in a disposable checkout. Do not use it as a per-experiment launch gate. |

The milestone number records terminal execution. It does not turn a blocked or
missing scientific result into a null result.

## The three largest PRD gaps

### Gap 1: local model text does not yet become a faithful constraint object

The PRD requires exact constraint extraction from real model outputs. Carnot
can preserve raw source and response bytes, but the completed V572 shards did
not test whether a model can identify the binding constraints before it solves.
There is no paired constraint-extraction result on a current local flagship.

V573 response:

1. Freeze an exact-checkable source set before inference.
2. Compare direct, always-on constraint-first, and routed constraint-first
   generation on matched units.
3. Preserve the plain-text constraint summary and final answer as separate raw
   receipts. Do not revive generated ConstraintIR.
4. Bind proposed constraints to source spans and let the exact checker decide
   final validity.

### Gap 2: no independently audited proof that staged constraints help

The architecture needs evidence that constraint extraction changes useful
behavior rather than formatting. V572 produced runtime receipts, but no
cross-family comparison survived. It also did not test source, stage, label,
or raw-byte counterfactuals.

V573 response:

1. Run Qwen3.6 and Gemma 4 31B in separate tasks and fresh processes.
2. Recompute every comparison from per-unit raw rows.
3. Measure exact success, constraint precision and recall, unsupported and
   contradictory constraints, abstention, tokens, latency, and failure rate.
4. Attack source binding, stage order, family labels, raw hashes, and exact
   authority before any method is adopted.

### Gap 3: continuous self-learning has no prospective safe result

FR-11 requires continuous self-learning. Exp6553 and Exp6554 blocked before
valid live transitions existed. Exp6552 showed that hysteresis was reversible,
but it did not beat matched controls. Carnot still lacks a chronological result
with retained support, poison resistance, restart, and rollback.

V573 response:

1. Keep all generator weights frozen.
2. Admit only independently exact-verified constraint rows.
3. Compare frozen memory, uniform replay, graph-Potts, protected-core plus
   occupancy, and conflict-routed specialist state.
4. Commit after the decision, evaluate on future rows, and require retention,
   future support, bounded growth, restart, rollback, and poison controls.

## Research inputs

The V573 source refresh is recorded in `research-references.md`.

- arXiv:2608.05254 introduces Constraint-First Reasoning. It separates a
  constraint-summary stage from a solve-and-check stage. It motivates the main
  paired experiment. Carnot imports the staged protocol and paired accounting.
  It does not import the paper's benchmark result.
- arXiv:2608.14569 argues that neural constraint solvers need instance-level
  symbolic certification when hard checks are cheap. It reinforces the existing
  exact-authority boundary.
- arXiv:2608.00220 shows that verifier-driven learning can improve current
  pass-at-one while reducing future reachable support. It motivates held
  future-support and retention gates for online memory.
- arXiv:2605.18871 keeps deterministic constraint energy separate from learned
  quality energy. V573 uses only the deterministic side for release.
- Extropic Z1 and Kona 1.0 remain strategic comparators. Neither provides an
  authenticated runner, public weights, or a reproducible local integration.

## Architecture

```text
V572 receipts + current exact registry + V573 source refresh
                           |
              +------------+-------------+
              |                          |
              v                          v
   Exp6585 terminal recovery    Exp6586 isolated suite truth
              |
              v
   Exp6587 constraint-first method contract
              |
       +------+------+
       |             |
       v             v
 Exp6588 Qwen     Exp6589 Gemma 31B
       |             |
       +------+------+
              v
   Exp6590 independent paired comparison
              |
              v
   Exp6591 counterfactual and authority audit
              |
       +------+------+
       |             |
       |      Exp6592 learner conformance
       |             |
       +------+------+
              v
   Exp6593 prospective continuous self-learning
              |
              v
   Exp6594 independent learning audit

   Exp6595 ARC redirect-ledger generalization
              |
              +-----------------------------+
                                            v
                               Exp6596 milestone capstone
```

The online reasoning path remains asymmetric.

```text
source bytes
    |
    v
plain-text proposed constraints ----> source-span binding
    |                                      |
    v                                      v
model final answer ----------------> exact obligation checker
                                           |
                               +-----------+-----------+
                               |                       |
                            release                 abstain
```

The model proposes. The exact checker certifies. Any gain defined by that
checker carries `verdict_class=circular_positive`, not `positive`.

## Phase I: evidence and method closure

### Exp6585 - V572 terminal recovery and V573 execution contract

Reconstruct all six V572 terminal rows and all three Exp6584 hard-limit
attempts. Freeze V573 time, checkpoint, cleanup, gate-field, and protected-file
contracts. This task loads no model.

**Acceptance:** every V572 terminal state replays from local evidence and the
new execution contract closes without inventing an Exp6584 artifact.

**Deliverable:** `results/experiment_6585_v573_terminal_recovery_and_execution_contract.json`

### Exp6586 - isolated repo-wide suite truth baseline

Run the full Python suite once in a disposable checkout with the project
coverage configuration. Compare tracked file hashes before and after. Record
all failures and mutation attempts. Do not make suite success a model-task gate.

**Acceptance:** the suite has one honest GREEN or RED baseline, the active
worktree is unchanged, and every failure and attempted tracked write is named.

**Deliverable:** `results/experiment_6586_isolated_full_suite_truth_baseline.json`

### Exp6587 - SOTA constraint-first method contract

Bind CFR and the retained controls to immutable primary-source receipts. Freeze
at least 16 exact-checkable units, direct and two-stage prompts, the restrictive
cue router, seeds, budgets, source-span binding, exact checkers, metrics, stop
rules, and attacks. Replay positive and negative fixtures without model load.

**Acceptance:** all fixtures replay and the exact field
`v573_constraint_first_method_ready_score` equals 1.

**Deliverable:** `results/experiment_6587_v573_constraint_first_method_contract.json`

## Phase II: paired local-model experiment

### Exp6588 - Qwen3.6 constraint-first stream

Run `unsloth/Qwen3.6-35B-A3B-GGUF` in one fresh llama.cpp process. Generate
matched direct, always-on constraint-first, and routed constraint-first rows.
Checkpoint each completed unit. The task makes no headline quality claim.

**Acceptance:** all expected Qwen rows, both CFR stages, exact results, token
costs, process receipts, and clean unload are independently recheckable.

**Deliverable:** `results/experiment_6588_qwen36_constraint_first_stream.json`

### Exp6589 - Gemma 4 31B constraint-first stream

Run `unsloth/gemma-4-31B-it-GGUF` under the same frozen contract in a separate
fresh process. Preserve every timeout and parse failure as a row.

**Acceptance:** the dense flagship satisfies the same row and runtime contract
as Exp6588 without family-specific prompt drift.

**Deliverable:** `results/experiment_6589_gemma4_31b_constraint_first_stream.json`

### Exp6590 - independent constraint-first comparison

Consume only completed raw artifacts. Recompute direct versus always-on and
routed CFR per unit and family. Report paired effects and uncertainty for exact
success, constraint quality, unsafe release, abstention, tokens, latency, and
failures. Do not trust upstream aggregates.

**Acceptance:** every aggregate replays. A positive candidate needs a strictly
positive routed-CFR exact-success delta, a nonnegative preregistered paired
lower bound, no unsafe-release increase, stage-one precision at or above the
frozen floor, and token cost within budget. An exact-defined win is circular.

**Deliverable:** `results/experiment_6590_constraint_first_independent_comparison.json`

### Exp6591 - constraint-first counterfactual and authority audit

Replay source-span replacement, constraint deletion, contradiction injection,
stage swap, family-label swap, raw-byte tamper, answer leak, and exact-authority
substitution attacks. This task performs no model inference.

**Acceptance:** every attack changes only the frozen expected links. Tamper,
leak, and authority substitution fail closed. The audit may pass when the
scientific comparison is null.

**Deliverable:** `results/experiment_6591_constraint_first_counterfactual_audit.json`

## Phase III: prospective continuous self-learning

### Exp6592 - protected conflict-memory conformance

Implement or harden graph-Potts state, protected trusted cores, per-source and
per-family occupancy, and bounded conflict specialists. Replay hand-computed
update, selection, conflict, poison, restart, and rollback fixtures. This task
makes no utility claim.

**Acceptance:** every fixture and invariant passes and
`learner_conformance_ready_score` equals 1.

**Deliverable:** `results/experiment_6592_protected_conflict_memory_conformance.json`

### Exp6593 - prospective exact-verified continuous self-learning

Use the audited constraint-first rows in chronological order. Compare frozen
memory, uniform exact replay, graph-Potts, protected-core plus occupancy, and
conflict-routed specialist arms. Snapshot before each decision. Commit only
after independent exact verification. Keep model weights frozen.

**Acceptance:** a candidate win must improve preregistered future support over
uniform replay while meeting retention, zero unsafe-commit, poison, occupancy,
memory, cost, restart, and rollback bounds. Otherwise report an honest null,
partial, block, or disqualification.

**Deliverable:** `results/experiment_6593_prospective_exact_verified_continuous_learning.json`

### Exp6594 - independent continuous-learning audit

Always run. Rebuild transitions, selections, conflicts, occupancies, retention,
future support, poison results, costs, restart, and rollback from rows and
journals. Do not import Exp6593 headlines.

**Acceptance:** all transitions and claims replay. Missing prospective input is
a named block, not a null result.

**Deliverable:** `results/experiment_6594_continuous_learning_independent_audit.json`

## Phase IV: ARC generalization and synthesis

### Exp6595 - ARC trajectory-supervisor generalization refinement

Read accumulated live `trajectory_supervisor` receipts with outcomes. Recompute
per-arm fired and helped counts by held game and time. Change only selection or
priority over the curated arm set when preregistered support exists. Do not add
a game adapter, inspect game source, run an outer-loop solve, or claim a level.
An empty ledger or unsupported change ends as `complete: no supported policy
change` and still satisfies the ARC generalization slot.

**Acceptance:** all ledger rows replay, any shared-policy change is backed by
held-game outcomes, and no per-game solve logic enters the live path.

**Deliverable:** `results/experiment_6595_arc_redirect_ledger_generalization_refinement.json`

### Exp6596 - independent V573 capstone

Account for every task and recompute all readiness and scientific dispositions.
Separate `positive`, `circular_positive`, `null`, `blocked`, `disqualified`, and
`partial`. Reconcile specs, traceability, status, and changelog with actual
evidence. Do not edit the active roadmap or conductor.

**Acceptance:** every expected task has a terminal artifact or exact missing
diagnosis. No block becomes null. No exact-defined gain becomes non-circular.

**Deliverable:** `results/experiment_6596_v573_independent_capstone.json`

## Dependency graph

```text
Exp6585 execution contract ---> Exp6588 Qwen -----+
             |                                    |
             +-----------------> Exp6589 Gemma ----+--> Exp6590 comparison
             |                                                   |
Exp6587 method contract --------+---------------------------------+
                                                                 v
                                                       Exp6591 audit
                                                                 |
Exp6592 learner conformance --------------------------------------+--> Exp6593 CSL
                                                                        |
                                                                        v
                                                                  Exp6594 audit

Exp6586 isolated suite ----------------------------------------------+
Exp6595 ARC generalization ------------------------------------------+--> Exp6596
Exp6585-Exp6594 -----------------------------------------------------+
```

Structured runtime gates:

- Exp6588 and Exp6589 require `v573_execution_contract_ready_score=1` and
  `v573_constraint_first_method_ready_score=1`.
- Exp6590 requires `qwen_constraint_first_rows_ready_score=1`,
  `gemma31_constraint_first_rows_ready_score=1`, and
  `v573_constraint_first_method_ready_score=1`.
- Exp6591 requires `constraint_first_comparison_rows_ready_score=1`.
- Exp6593 requires `constraint_first_audit_ready_score=1` and
  `learner_conformance_ready_score=1`.

Exp6586, Exp6592, Exp6594, Exp6595, and Exp6596 always run. They can diagnose
missing input without disappearing behind a gate cascade.

## Model requirements

| Task | Required local model | Role |
|---|---|---|
| Exp6588 | `unsloth/Qwen3.6-35B-A3B-GGUF` | Flagship MoE matched constraint-first stream |
| Exp6589 | `unsloth/gemma-4-31B-it-GGUF` | Flagship dense matched constraint-first stream |

`unsloth/gemma-4-26B-A4B-it-GGUF` remains an allowed SOTA replication family
when capacity permits. Legacy Qwen3.5-0.8B and gemma-4-E4B-it may smoke-test CPU
plumbing only. They cannot satisfy a readiness field or headline result.

Each live task follows `cached_sota_pair()` cache-resolution practice, records
content-derived GGUF identity, uses the embedded tokenizer through llama.cpp,
runs one model at a time, samples GPU state, checkpoints raw rows, and verifies
unload. It never calls `AutoTokenizer` on a GGUF repository.

## Hardware requirements

| Resource | Tasks | Contract |
|---|---|---|
| RTX 3090 GPU 0 | Exp6588 and Exp6589 | Conductor-owned. Run one model task at a time. Preserve unrelated processes. Record offload, memory, utilization, timing, and unload. |
| RTX 3090 GPU 1 | none planned | Outer-loop owned. Do not claim or kill its active process. |
| CPU, RAM, and local disk | All tasks | Record preconditions. Keep raw rows content-addressed. Use atomic terminal artifacts. |
| Temporary disposable checkout | Exp6586 | Run the repo-wide suite outside the active worktree. Compare tracked hashes before and after. |
| KV260 | none | Terminal state is already adversarial-verified. No new command or performance claim. |
| GateMate | none | Terminal state is already reached. No command without a new operator physical receipt. |
| PolarFire | none | Terminal workload validation is already recorded. No repeat smoke. |
| Extropic TSU | none | No authenticated runner exists. Keep watch-only. |

No hardware purchase is required. The hardware wishlist remains the source of
truth for future acquisitions.

## Failed-scope discipline

| V573 task | Prior terminal scope | Material change | Repeat rule |
|---|---|---|---|
| Exp6585 | Exp6584 hard limit with no artifact | Receipt-only recovery; no model or aggregate science | Retire this recovery shape if it also ends with the same no-artifact hard limit. |
| Exp6587 | Exp6528 source-method contract blocked on OpenReview metadata | Direct arXiv receipts and local exact fixtures; no OpenReview date gate | Retire this source-contract shape if the same unavailable-channel verdict repeats. |
| Exp6588 | Exp6581 blocked before Qwen load | Full suite removed from launch checks; new CFR paired output; per-unit checkpoints | Retire this Qwen CFR runtime scope if the same precondition block repeats. |
| Exp6590 | Exp6584 monolithic three-family audit timed out | Two completed artifacts only; no model load; deterministic reducer | Retire this reducer shape if it repeats the same hard-limit/no-artifact result. |
| Exp6593 | Exp6553 blocked on GPU and live receipt preconditions | No model load; audited immutable rows; chronological reversible memory | Retire if the same missing-live-evidence block repeats. |
| Exp6594 | Exp6554 blocked on missing learning inputs | Audits current journals and transitions and always runs | Retire if the same missing-input audit verdict repeats. |
| Exp6595 | Exp6558 found reachability but no supported policy change | Replays newer outcome-bearing ledgers and freezes a held-game support rule | Retire this priority-refinement method if the same no-change verdict repeats. |

Each corresponding YAML task carries all four required `prior_failures` fields,
including `retire_if_same_verdict: true`.

## Milestone acceptance

V573 succeeds operationally when all 12 tasks reach terminal artifacts or
structured pre-gate terminal records and Exp6596 accounts for every task.

The constraint-first result is eligible only when:

1. both flagship streams pass independent row replay;
2. prompts, sources, stages, and model identities remain frozen;
3. exact results and costs recompute per unit;
4. counterfactual and authority attacks fail closed;
5. the verdict preserves exact-checker circularity.

The continuous-learning result is eligible only when:

1. inputs are independently audited and chronological;
2. generator weights remain frozen;
3. every commit follows an exact verification receipt;
4. retention and future support use held future rows;
5. poison, occupancy, memory, cost, restart, and rollback checks pass.

Honest terminal outcomes include no CFR gain, a gain that costs too many
tokens, low constraint precision, no online-memory benefit, an empty ARC
redirect ledger, or no supported ARC priority change. Those are null findings
when the full protocol ran. A missing input is blocked.

## Required reconciliation

Before implementation changes, each task must confirm a relevant capability
spec with `REQ-*` and `SCENARIO-*` anchors. Tasks write focused tests first and
run the applicable unit, lint, spec-coverage, artifact, adversarial, and E2E
checks from `ops/e2e-test-plan.md`.

Exp6596 reconciles:

- relevant `openspec/capabilities/*/spec.md` files and this proposal;
- `_bmad/traceability.md`;
- `_bmad/architecture.md` only if the accepted architecture changes;
- `ops/status.md`;
- `ops/changelog.md`;
- `ops/exclusion_manifest.yaml` for repeated retired verdicts.

Protected throughout the milestone:

- `research-roadmap.yaml`;
- `scripts/research_conductor.py`;
- prior terminal artifacts;
- operator-owned GPU processes and dirty worktree changes.

Do not push.

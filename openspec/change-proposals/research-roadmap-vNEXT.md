# Research Roadmap V564: Transactional Exact Guidance and Adaptive Self-Learning

**Milestone:** 2026.08.564  
**Date:** 2026-08-22  
**Status:** Planned  
**Experiments:** Exp6513-Exp6526  
**Execution file:** `research-roadmap-next.yaml`

## Purpose

V564 tests one narrow hypothesis:

> A crash-safe exact branch dataset can expose causal structural headroom, and
> a refinement-checked conflict memory can improve later exact queries without
> changing answers, hiding regressions, or weakening native fallback.

The milestone does not treat terminal execution as scientific success. It fixes
the repeated artifact transaction failure before it repeats the branch-data
work. It uses learned advice only after an independent structural certificate.
It does not reopen answer-level energy scoring, external-text verification,
hidden-state scoring, factor spawning, offline ARC solving, or a hardware speed
claim.

## What V563 proved

V563 reached terminal records for Exp6510-Exp6512. Its planned scientific chain
did not run.

1. **The immutable exact evidence is still usable.** The Exp6510 file contains
   480 checked rows, stable input receipts, a `partial` class, and
   `v563_independent_root_ready_score=1.0`.
2. **The execution contract failed again.** The conductor returned
   `artifact_not_updated_past_bootstrap` three times for Exp6510 and retired the
   task even though content later existed at the deliverable path. This is the
   same terminal failure class as Exp6506.
3. **The dataset did not run.** Exp6511 was skipped because its structured
   upstream was retired. No Exp6511 deliverable exists.
4. **The audit failed closed.** Exp6512 ran without a structured gate, found
   zero data rows and an incomplete manifest, wrote
   `branch_dataset_audited_ready_score=0.0`, and reported `verdict_class=blocked`.
5. **No method claim opened.** V563 did not prove branch headroom, learned
   routing value, exact-conflict reuse value, or continuous self-learning.

The proper handoff is therefore a direct, hash-bound read of immutable files,
not a dependency on a retired task ID. The first implementation deliverable in
V564 is a reusable atomic shard-and-final transaction with recovery tests.

## Three largest gaps to the PRD vision

| Gap | Current evidence | V564 response |
|---|---|---|
| Research execution is not crash-safe enough for long exact-data tasks | Exp6506 and Exp6510 both ended `artifact_not_updated_past_bootstrap`; both downstream dataset tasks were retired or skipped | Add a tested artifact transaction, recovery journal, content-addressed shards, atomic final replace, and closed failure artifact before data generation |
| Carnot has no causal held evidence that learned guidance helps an exact solver | Exact labels exist, but branch counterfactuals and structural controls never ran | Build a bounded pilot, independently audit it, compare native and exact structural controls, then gate a Safety Net router on held headroom |
| FR-11 continuous self-learning has no sound held-future benefit | Exp6496 was row-complete but null; Exp6498 confirmed held-future benefit failed | Replace factor proposals with refinement-valid exact conflicts, transactional memory, chronological replay, rollback, bounded eviction, and adaptive validation backed by an immutable full audit |

The ARC transfer gap remains a standing requirement. V564 reserves one task for
redirect-ledger supervisor refinement. The task reads live-path outcomes and
makes no game or level solve claim.

## Research findings used by V564

The full dated source refresh is in `research-references.md`.

- **Task-CoEvolve, arXiv:2608.20169:** variance-weighted task selection can
  concentrate validation on the capability frontier while probability-aware
  estimates represent the full set. Exp6523 compares full, fixed-subset, and
  adaptive validation around the chronological conflict-memory stream. A
  frozen full audit and exact sentinel set remain mandatory.
- **Safety Nets, arXiv:2608.20053:** a compact learned fast path plus a lookup
  table for residual errors can reduce storage over a bounded discrete domain.
  Exp6520 adapts the pattern to branch ordering with a content-hashed exception
  table and native exact fallback. Advice never prunes candidates or certifies
  answers.
- **Incremental verification through learned conflicts, arXiv:2603.12232:**
  conflict reuse is sound only under a proved refinement relation. Exp6521 and
  Exp6522 make that witness the durable-write gate.
- **DiBS, arXiv:2606.06518:** learned global structure can order values while a
  complete solver retains all candidates. Exp6518 includes consistency-aware
  ordering as a control before any learned router.
- **Nested SMC for discrete diffusion, arXiv:2608.20123:** the method is a
  future decoder control, not a current autoregressive GGUF dependency.
- **ChainForge, arXiv:2608.15961:** embedding and remapping can dominate an
  annealing workload. V564's fixed-width record reports logical and mapped
  sizes, but it makes no board execution, acceleration, latency, or power claim.
- **Ferrotherm:** its device trait, exact small-system checks, topology mapping,
  and joules-ledger shape are implementation references only. V564 does not
  import the repository or repeat its claims.

Current OpenReview, Hugging Face, Semantic Scholar, Extropic, Kona, KAN, Ising,
and GitHub checks do not change the exact-authority boundary. Extropic still
describes Z1 access in 2027. Kona still has no public local runner.

## Scientific invariants

1. The installed exact SAT or CSP solver owns labels, accepted solutions, and
   release decisions.
2. Advice may order variables or values, request bounded refocus, abstain, or
   fall back. It may not prune a candidate or accept a result.
3. Every admitted arm returns the same exact answer on each unit.
4. Splits are sealed by base-instance lineage. Held rows cannot be repaired
   after a held result is read.
5. Every comparative task emits one row per unit, arm, seed, shift, budget, and
   terminal disposition.
6. Feature, training, lookup, exception-table, solver, fallback, mapping, and
   validation-selection costs are charged.
7. Exact solver self-checks are `circular_positive` at most. They cannot create
   an oracle-distinct method claim.
8. Conflict memory persists only facts with a refinement witness and exact
   replay receipt. Failed writes roll back to the prior content hash.
9. Adaptive validation never controls validity or release. A frozen exact
   sentinel runs every iteration, and the full held set runs at the end.
10. A blocked artifact names the failed field, expected value, observed value,
    and source path in `gate_check_summary`.
11. No structured gate or `requires` chain names Exp6506, Exp6507, Exp6508,
    Exp6509, Exp6510, or Exp6511.
12. Historical artifacts, `research-roadmap.yaml`, and
    `scripts/research_conductor.py` remain unchanged.
13. Exp6524 reports supervisor-selection evidence only. It cannot claim an ARC
    game or level solve.
14. Exp6525 runs no GateMate command unless a new dated physical-state receipt
    exists after Exp6325. An unchanged state ends as a documented block.

## Local model policy

The credited path is procedural exact solving, local feature models, artifact
replay, and live-path receipt reduction. No planned experiment needs an LLM.

If implementation adds an LLM arm, its `MODEL_SPECS` must call
`cached_sota_pair()` and include at least one of:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The arm must load the returned GGUF `model_path` through `llama_cpp` and use the
GGUF-embedded tokenizer. Qwen3.5-0.8B and gemma-4-E4B-it are CPU smoke tests
only and cannot support a headline. Exp6524 reads recorded live receipts. It
does not invoke an ARC generator.

## Architecture

```text
Immutable direct inputs by path + hash
  Exp6504 exact rows + Exp6506/6510 receipts + Exp6512 terminal block
                           |
                           v
              Exp6513 terminal handoff contract

Exp6514 atomic shard transaction ---- Exp6515 literature/method contract
                    \                    /
                     v                  v
                  Exp6516 bounded branch pilot
                              |
                              v
                  Exp6517 independent pilot audit
                         /                    \
                        v                      v
          Exp6518 structural controls   Exp6521 conflict-memory controller
                        |                      |
                        v                      v
          Exp6519 headroom certificate  Exp6522 chronological self-learning
                        |                      |
                        v                      v
          Exp6520 Safety Net router      Exp6523 adaptive validation + audit

Exp6524 ARC redirect-ledger generalization ----\
Exp6525 GateMate changed-state continuity ------> Exp6526 V564 capstone
All terminal records --------------------------/
```

The exact solver remains on the acceptance path. The learned router is a
sidecar. The exception table and conflict memory are content-addressed exact
stores, not new release authorities.

## Phase A: Repair execution and establish audited data

### Exp6513 - V564 terminal handoff contract

Recompute the V563 terminal record from direct immutable paths. Record the two
bootstrap-update failures, the usable but retired Exp6510 content, the missing
Exp6511 deliverable, and the closed Exp6512 block. This is a governance record,
not another exact-root qualifier.

**Deliverable:**
`results/experiment_6513_v564_terminal_handoff_contract.json`

**Gate:** `v564_handoff_ready_score == 1.0` is informational. No later task has
a structured dependency on it.

### Exp6514 - Atomic shard-and-final artifact transaction

Specify, test, and implement a reusable transaction for long experiments. It
writes content-addressed row shards, journals planned and terminal unit IDs,
verifies resume hashes, atomically replaces the final JSON, and writes a closed
failure artifact when finalization cannot complete. It does not modify the
conductor.

**Deliverable:**
`results/experiment_6514_atomic_shard_artifact_transaction.json`

**Gate:** `atomic_artifact_contract_ready_score == 1.0`.

### Exp6515 - Execution-time literature and method contract

Run a focused low-concurrency source delta. Freeze the V564 method contract for
Task-CoEvolve, Safety Nets, learned conflicts, DiBS, and ChainForge. Record
implementable methods, non-transferable claims, exact-authority boundaries, and
source hashes. This task cannot widen V564 after later results are visible.

**Deliverable:**
`results/experiment_6515_v564_source_method_contract.json`

**Gate:** `v564_method_contract_ready_score == 1.0`.

### Exp6516 - Bounded exact branch-counterfactual pilot v3

Read immutable Exp6504 and Exp6510 content by path and hash. Do not use their
task IDs as structured dependencies. Generate a bounded pilot with sealed
checkpoints, all eligible values, equal exact budgets, explicit censoring,
content-addressed shards, and a terminal manifest through Exp6514's transaction.

**Deliverable:**
`results/experiment_6516_exact_branch_pilot_dataset_v3.json`

**Gate:** `branch_pilot_dataset_ready_score == 1.0`.

### Exp6517 - Independent branch-pilot audit

Always run. Recompute rows, solver receipts, terminal dispositions, shard
hashes, split lineage, feature timing, censoring, and shortcut attacks. Missing
or invalid input produces a complete blocked artifact and a closed zero score.

**Deliverable:**
`results/experiment_6517_branch_pilot_independent_audit.json`

**Gate:** `branch_pilot_audited_ready_score == 1.0`.

## Phase B: Prove headroom before learned routing

### Exp6518 - Structural-control headroom A/B v2

Compare native dynamic branching, shuffled order, static analytical order,
partial-assignment consistency order, bounded periodic refocus, random
critical-variable enumeration, and analytical enumeration. Use matched budgets,
charge all work, and record whether advice changes a live branch decision.

**Deliverable:**
`results/experiment_6518_structural_control_headroom_ab_v2.json`

### Exp6519 - Independent structural-headroom certificate

Always run. Recompute Exp6518 from rows and reject correctness drift, inert
advice, identity shortcuts, omitted hard units, uncharged work, one-cell wins,
or unsupported transfer from papers. Emit the only gate that can open learned
routing.

**Deliverable:**
`results/experiment_6519_structural_headroom_certificate.json`

**Gate:** `certified_structural_headroom_score == 1.0`.

### Exp6520 - Safety Net branch-router A/B

Only run after certified structural headroom. Compare native, analytical,
linear, MLP, and compact KAN routers under matched feature and training budgets.
Pair each learned arm with a content-hashed exception table, abstention, full
candidate preservation, and native exact fallback. Exhaustively audit the
bounded pilot domain.

**Deliverable:**
`results/experiment_6520_safety_net_branch_router_ab.json`

## Phase C: Continuous self-learning through exact conflict memory

### Exp6521 - Transactional refinement-witness conflict memory

Implement a versioned exact-conflict store. Admit a conflict only when a proved
query-refinement relation and exact replay show it remains valid. Include
invalid-reuse veto, deterministic capacity and eviction, commit, rollback,
restart, corruption quarantine, native fallback, and a fixed-width mapping
record with logical and mapped sizes.

**Deliverable:**
`results/experiment_6521_transactional_refinement_conflict_memory.json`

**Gate:** `conflict_memory_controller_ready_score == 1.0`.

### Exp6522 - Chronological exact-conflict self-learning

Run a sealed stream of related and unrelated queries. Compare scratch, frozen
memory, valid reuse, bounded valid reuse, restart, rollback, and invalid-reuse
attack arms. Measure immediate cost, old-prefix retention, held-future support,
interference, durable writes, and exact answer equality.

**Deliverable:**
`results/experiment_6522_chronological_conflict_self_learning.json`

This is the milestone's required continuous self-learning experiment.

### Exp6523 - Adaptive validation and independent self-learning audit

Independently replay Exp6522. Compare full-set validation, a fixed subset, and
Task-CoEvolve-style variance-weighted sampling with inverse-probability
estimates. Run the same immutable exact sentinel every iteration and the full
held set at the end. Adaptive validation may reduce checks. It cannot change
the self-learning acceptance rule.

**Deliverable:**
`results/experiment_6523_adaptive_validation_csl_audit.json`

**Gates:** `adaptive_validation_ready_score == 1.0` supports an evaluation-cost
claim. `continuous_self_learning_claim_eligible_score == 1.0` separately
supports an FR-11 claim.

## Phase D: Generalization, hardware continuity, and synthesis

### Exp6524 - ARC redirect-ledger supervisor generalization

Read live `trajectory_supervisor` receipts carrying `arm_outcomes`. Recompute
fired, helped, actions-to-progress, and unredirected-stagnation rows. Retire or
reprioritize an arm only when precommitted evidence supports the change. If no
arm fired, report `no_firings_nothing_to_refine`. Do not generate an arm, run an
offline solve, or claim a game or level solve.

**Deliverable:**
`results/experiment_6524_arc_supervisor_redirect_generalization.json`

### Exp6525 - GateMate changed-state continuity

Satisfy the still-open board-continuity rule without repeating an unchanged
JTAG probe. Look only for a new dated operator receipt after Exp6325. If none
exists, run zero hardware commands and record the external physical action that
remains. If one exists, allow one bounded detect or flash step and stop at its
first terminal result. Make no speed, latency, energy, or availability claim.

**Deliverable:**
`results/experiment_6525_gatemate_changed_state_continuity.json`

KV260 and PolarFire already have terminal artifacts for the continuity rule:
Exp3600 records the KV260 terminal latency transcript with synthesis success,
and Exp5347 records a hash-validated PolarFire workload. They need no V564 slot.

### Exp6526 - Independent V564 capstone

Recompute every claim from per-unit rows. Audit retired-scope isolation, atomic
finalization, exact authority, split leakage, charged cost, gate spelling,
verdict class, continuous-learning retention, adaptive-validation bias, ARC
provenance, GateMate continuity, and protected-file hashes. Reconcile specs and
ops documents only after the evidence table is frozen.

**Deliverable:**
`results/experiment_6526_v564_independent_capstone.json`

## Dependency graph

```text
6514 atomic transaction == 1 ----\
                                    > 6516 pilot -> 6517 audit == 1
6515 method contract == 1 --------/                     |
                                                          +-> 6518 controls
                                                          |      |
                                                          |      v
                                                          |   6519 certificate == 1
                                                          |      |
                                                          |      v
                                                          |   6520 Safety Net router
                                                          |
                                                          +-> 6521 conflict controller == 1
                                                                 |
                                                                 v
                                                              6522 CSL complete == 1
                                                                 |
                                                                 v
                                                              6523 adaptive audit

6513 handoff, 6524 ARC, and 6525 GateMate run independently.
6526 reads every terminal record and never cascade-blocks.
```

Structured conductor gates exist only where a failed prerequisite makes the
downstream implementation meaningless. Independent audits and the capstone run
without structured gates so every gate field closes even on missing input.

## Hardware requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| CPU and RAM | Exp6513-Exp6519, Exp6521-Exp6526 | Exact replay, artifact hashing, solver controls, conflict memory, audits, ARC receipt reduction, and GateMate state audit |
| Dual RTX 3090 | Exp6520 only if its compact learned arms need CUDA | Optional training acceleration; CPU parity and total charged cost remain reported; no GPU speedup headline |
| Local GGUF cache | None on the credited path | If an unplanned LLM arm is added, use `cached_sota_pair()` and at least one mandated SOTA GGUF; legacy small models are smoke-only |
| GateMate A1-EVB-2M | Exp6525 | No command without a post-Exp6325 dated physical-state receipt; at most one bounded step; no performance claim |
| KV260 | No V564 task | Terminal continuity evidence already exists in Exp3600 |
| PolarFire | No V564 task | Terminal hash-validated workload evidence already exists in Exp5347 |
| TSU or Kona hardware | None | No authenticated local route; context only |

The conflict record includes a fixed-width hardware mapping and reports logical
records, physical records, topology expansion, and mapping time. It is a CPU
reference ABI, not an FPGA or thermodynamic execution result.

## Acceptance and stop rules

V564 may support a learned-router claim only if Exp6519 certifies held
structural headroom and Exp6520 preserves exact answers, all candidates,
abstention, fallback, exhaustive pilot coverage, and charged positive benefit.

V564 may support a continuous self-learning claim only if Exp6523 independently
confirms:

- zero unsafe or invalid durable writes;
- exact answer equality for every arm and unit;
- positive held-future benefit after charged lookup and validation cost;
- old-prefix retention within the precommitted margin;
- support preserved under bounded capacity, restart, rollback, and corruption;
- benefit beyond scratch, frozen, and matched-dose controls;
- the same conclusion under the immutable full-set audit.

Stop the learned-router line at a null Exp6519. Stop the self-learning line at a
null Exp6523. Do not convert either null into a larger rerun inside V564. A
missing prerequisite produces a closed blocked artifact with the named gate
value. A repeated terminal verdict on a declared rerun retires the new task ID.

## Deliverable index

| Exp | Deliverable | Class |
|---|---|---|
| 6513 | `results/experiment_6513_v564_terminal_handoff_contract.json` | Infrastructure/governance |
| 6514 | `results/experiment_6514_atomic_shard_artifact_transaction.json` | Infrastructure |
| 6515 | `results/experiment_6515_v564_source_method_contract.json` | SOTA ingestion |
| 6516 | `results/experiment_6516_exact_branch_pilot_dataset_v3.json` | Data |
| 6517 | `results/experiment_6517_branch_pilot_independent_audit.json` | Infrastructure/audit |
| 6518 | `results/experiment_6518_structural_control_headroom_ab_v2.json` | Structural experiment |
| 6519 | `results/experiment_6519_structural_headroom_certificate.json` | Independent audit |
| 6520 | `results/experiment_6520_safety_net_branch_router_ab.json` | Learned guidance |
| 6521 | `results/experiment_6521_transactional_refinement_conflict_memory.json` | Self-learning mechanism |
| 6522 | `results/experiment_6522_chronological_conflict_self_learning.json` | Continuous self-learning |
| 6523 | `results/experiment_6523_adaptive_validation_csl_audit.json` | Adaptive validation/audit |
| 6524 | `results/experiment_6524_arc_supervisor_redirect_generalization.json` | ARC generalization |
| 6525 | `results/experiment_6525_gatemate_changed_state_continuity.json` | Hardware continuity |
| 6526 | `results/experiment_6526_v564_independent_capstone.json` | Independent synthesis |

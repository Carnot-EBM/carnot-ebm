# Carnot Research Roadmap vNEXT: Provenance-Complete Constraint Routing and Unique-Event Self-Learning

**Created:** 2026-08-15

**Milestone:** 2026.08.556

**Status:** Planned

**Supersedes:** milestone 2026.08.555, experiments 6448-6459

**Primary evidence:** V555 artifacts, `ops/conductor-log.md`,
`ops/exclusion_manifest.yaml`, and the V556 source refresh in
`research-references.md`

The architecture document was last reconciled on 2026-07-03. It is stale under
the repository's 30-day rule. This roadmap uses it only for stable component
names. V555 artifacts and current operational records define live behavior.

## 1. What milestone 2026.08.555 proved

V555 completed its activated queue. It repaired the path-receipt boundary and
ran two nominal continuous-learning experiments. Its independent audits did
not grant a science claim.

| Area | V555 evidence | What it proves | Remaining limit |
|---|---|---|---|
| Queue | Exp6448 set `v555_queue_integrity_score=0`. It found missing prior-failure declarations. | Preflight can fail closed and name an activation defect. | V556 must declare every matching prior scope before activation. |
| Path receipts | Exp6449 set `path_receipt_ready_score=1`. | Raw bytes, parser state, checker transport, and verdict can be bound and localized. | The receipt must now guard live SOTA writes, not fixtures only. |
| SOTA corpus | Exp6450 ran all three mandated GGUFs for 2,013.6 seconds. It set `sota_corpus_ready_score=0`. Of 324 candidate rows, 182 had missing or zero-byte raw output, and 269 candidate rows were duplicates. | The local runners work, but the corpus is not a valid evidence substrate. | No grounding, allocation, or energy claim may use this corpus. |
| Constraint-routing chain | Exp6451 and Exp6453 were gate-blocked. Exp6452 and Exp6454 were pre-gate skips with no artifacts. | The conductor preserved the failed corpus gate. | Carnot still has no causal exact-decision gain from grounding or energy. |
| Continuous learning | Exp6455 and Exp6456 reported positive exact effects. | The external factor-weight mechanism can run, persist, roll back, and restart. | Exp6457 found only 72 unique raw outputs in 216 rows and an exact-veto attack that did not fail closed. The claims are ineligible. |
| ARC | Exp6458 reduced some collisions and improved reachability. | Representation and objective changes can alter live-path metrics without a solve claim. | The frozen safety roster regressed on `g50t`, and the capstone found two row-aggregate mismatches. |
| Capstone | Exp6459 completed an independent audit and granted no requested science, public ARC, or hardware claim. | The release boundary held. | The PRD gaps remain open. |

The key lesson is causal. A positive aggregate is not evidence when raw events
are missing, cloned, or admitted before the exact veto. V556 repairs those
contracts before it retries the scientific questions.

## 2. The three largest gaps to the PRD vision

### Gap 1: Carnot lacks a provenance-complete SOTA evidence substrate

The PRD requires reproducible local inference and deterministic validation.
V555 produced live SOTA rows, but more than half lacked nonzero raw bytes and
most candidate rows were duplicates. V556 must prove atomic raw persistence,
one event identity per generation, sealed partitions, and exact held headroom
before downstream work runs.

### Gap 2: Exact constraints have not caused a held final-decision gain

Carnot has exact checkers and admitted factors. It still lacks a held result in
which fixed grounding, an explicit objective, or constraint energy changes the
selected action and improves exact success. V556 separates four questions:
fact grounding, objective choice, verifier allocation, and final energy
selection. Each stage uses matched bytes and a sealed held split.

### Gap 3: FR-11 has no independently eligible continuous-learning result

V555's nominal learning gains came from reused generations and an open exact-
veto attack. V556 uses one unique raw generation per learning event. The exact
checker runs before write admission. A frozen arm, corrupt-feedback arm,
rollback, restart, held chronology, and an independent reducer test the full
FR-11 loop.

ARC remains a secondary live-path risk. V556 adds a generic safety shield and
canonical reducer. It makes no public game or level solve claim.

## 3. Research hooks from the V556 refresh

| Source | V556 hook | Boundary |
|---|---|---|
| Beyond Final Scores, arXiv:2608.13417 | Score generation, parse, verify, write, restart, and held use as separate events. | A terminal score cannot repair a broken process path. |
| LittleLearner, arXiv:2608.13545 | Use an exposure ledger to distinguish acquisition from reuse. | A held unit or raw output seen during updates is not held evidence. |
| AutoDesign, arXiv:2608.13560 | Test recursive improvement only behind a frozen validation boundary. | The learner may not edit the conductor, checker, split, or release gate. |
| CrEST, arXiv:2608.13179 | Let exact outcomes choose update direction. | Model evidence may scale magnitude only. |
| Objective Is the Bottleneck, arXiv:2608.12959 | Freeze representations while objectives change. | Probe quality alone is not a release claim. |
| Sampling Luck Masquerades as Allocation Gain, arXiv:2608.13087 | Freeze allocation on development units and test at equal held cost. | Include harmful-flip and zero-gain controls. |
| Policy-as-logic, arXiv:2608.11905 | Ground into fixed typed slots, then run fixed rules. | Do not generate a new constraint language or solver. |

Semantic Scholar still returned 33 EBT citing records and eight ARM-EBM
citing records. No new citation changes the order below. Extropic exposes no
authenticated local TSU route. Kona exposes no public local runner. KAN and
Ising work do not close a current evidence gap.

## 4. V556 architecture

```text
                             exact authority boundary
                                      │
fixed policy unit                     │
      │                               │
      ▼                               │
┌──────────────────────────────┐      │
│ Mandatory local GGUF runner  │      │
│ one generation → one event   │      │
└──────────────┬───────────────┘      │
               │ atomic nonzero bytes │
               ▼                      │
┌──────────────────────────────┐      │
│ Event + path receipt         │      │
│ model/file/tokenizer/device  │      │
│ prompt/raw/parser/checker    │      │
└──────────────┬───────────────┘      │
               │                      │
               ▼                      │
┌──────────────────────────────┐      │
│ Fixed typed grounding        │      │
│ fixed rule program           │      │
└──────────────┬───────────────┘      │
               │                      │
               ▼                      │
┌──────────────────────────────┐      │
│ Objective + budget + energy  │      │
│ proposal and routing only    │      │
└──────────────┬───────────────┘      │
               │ selected candidate   ▼
               └──────────────────▶ ┌────────────────────┐
                                    │ Exact local checker │
                                    └─────────┬──────────┘
                                              │ pass / veto
                  ┌───────────────────────────┴──────────┐
                  ▼                                      ▼
          release or abstain                   immutable event log
                                                         │
                                                         ▼
                                              ┌────────────────────┐
                                              │ bounded factor     │
                                              │ update + rollback  │
                                              └────────────────────┘

Independent ARC lane:
live runtime traces → frozen objective → generic safety shield → canonical
row reducer → reachability and safety metrics; no source read and no solve
```

Only the deterministic local checker can authorize release or a learning
write. Learned or heuristic components may propose, rank, route, or abstain.

## 5. Phase 0 - Handoff, source delta, and raw-event substrate

### Exp6460 - V555 handoff and V556 queue integrity

Freeze all V555 terminal determinations. Validate 13 V556 tasks, gate producer
fields, prompt endings, model policy, exclusions, and every prior-failure
block. This task is infrastructure. No science task depends on it.

Deliverable:
`results/experiment_6460_v556_terminal_handoff_and_queue_integrity.json`

### Exp6461 - V556 SOTA source and benchmark delta receipt

Recheck the latest arXiv release, EBT and ARM-EBM citation trails, OpenReview,
Hugging Face Papers, GitHub, Extropic, Logical Intelligence, and the rendered
ARC leaderboard. Preserve source timestamps and primary links. This is a
source-ingestion task, not a product or benchmark claim.

Deliverable:
`results/experiment_6461_v556_sota_source_and_benchmark_delta.json`

### Exp6462 - Atomic raw-output persistence and uniqueness canary

Run a small live matrix across all three mandated GGUFs. Every generation gets
a new path, atomic write, nonzero byte check, content hash, event ID, model and
device receipt, and replay through Exp6449's path chain. Inject zero-byte,
duplicate-event, and candidate-clone attacks. This is a changed recovery of
Exp6450's evidence failure.

Deliverable:
`results/experiment_6462_sota_raw_persistence_uniqueness_canary.json`

### Exp6463 - Provenance-complete SOTA fixed-policy corpus v2

Rebuild the corpus only if Exp6462 passes. Use all three mandatory GGUFs. Seal
development, allocation-held, selection-held, and audit-held partitions. Each
generation must have unique nonzero raw bytes. Every held partition must have
mixed exact outcomes and selection headroom.

Deliverable:
`results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json`

## 6. Phase 1 - Causal constraint routing

### Exp6464 - Fixed-slot grounding with exact rule execution

Consume Exp6463 bytes without new inference. Compare policy-as-prompt,
fixed-slot policy-as-logic, an ablated grounding, and a gold-fact upper bound.
Use a fixed predicate inventory and fixed solver. Measure false accepts and
exact task success per unit.

Deliverable:
`results/experiment_6464_fixed_slot_grounding_exact_logic_ab.json`

### Exp6465 - Representation-versus-objective causal A/B v2

Freeze Exp6464 facts and candidates. Compare the current violation sum, a
lexicographic protected-clause-first objective, an ablation, and a shuffled
placebo. Require a row-level active-versus-ablated effect before readiness.

Deliverable:
`results/experiment_6465_representation_objective_causal_ab_v2.json`

### Exp6466 - Held verifier allocation at equal total cost v2

Freeze an allocation policy on development rows. Compare it with uniform,
always-check, shuffled zero-gain, and in-sample oracle diagnostics on the
allocation-held split. Charge every probe and report harmful flips.

Deliverable:
`results/experiment_6466_held_verifier_budget_allocation_v2.json`

### Exp6467 - Held exact-constraint energy selection v2

Freeze the Exp6465 objective before opening selection-held rows. Compare first
candidate, vote, shuffled energy, violation sum, and the changed objective.
Final exact success is the headline. The energy score is never authority.

Deliverable:
`results/experiment_6467_held_exact_constraint_energy_selection_v2.json`

## 7. Phase 2 - Unique-event continuous self-learning

### Exp6468 - Prospective unique-event verifier-bounded learning

This is the required continuous self-learning task. It is independent of the
Phase 1 corpus chain. Generate a fresh chronological stream with all three
mandatory GGUFs. One live generation is one learning event. The exact checker
runs before write admission. Compare frozen weights, self-signed updates, and
exact-sign verifier-bounded updates. Updates affect future units only.

Deliverable:
`results/experiment_6468_unique_event_verifier_bounded_csl.json`

### Exp6469 - Corrupt feedback, rollback, and held restart replication v2

Run new held generations after a real process restart. Inject forged pass,
replayed output, wrong-unit binding, and checker-transport corruption. The
governed learner must quarantine, tombstone, roll back, and prevent
resurrection before any protected release.

Deliverable:
`results/experiment_6469_unique_event_csl_corruption_restart.json`

### Exp6470 - Independent CSL provenance and lifecycle audit v2

Remain ungated. Recompute Exp6468 and Exp6469 from raw files and event logs.
Check one-to-one event identity, exposure chronology, exact-veto ordering,
rollback, restart, held disjointness, row aggregates, and duration. Grant
eligibility only if all critical attacks fail closed.

Deliverable:
`results/experiment_6470_independent_unique_event_csl_audit.json`

## 8. Phase 3 - ARC safety and capstone

### Exp6471 - Generic ARC safety-shield objective A/B

Use runtime traces only. Freeze the best Exp6458 representation and objective.
Add a generic safety veto and conservative fallback. Evaluate leave-one-game-
out reachability, legal actions, and the frozen safety roster with a canonical
row reducer. Do not read game source. Make no game or level solve claim.

Deliverable:
`results/experiment_6471_arc_generic_safety_shield_objective_ab.json`

### Exp6472 - V556 independent adversarial capstone

Recompute every requested headline from rows and raw evidence. Test gate
contracts, path receipts, event uniqueness, held contamination, exact-veto
ordering, aggregate consistency, ARC provenance, and protected files. Reconcile
spec and operations documents only from eligible determinations.

Deliverable:
`results/experiment_6472_v556_adversarial_capstone.json`

## 9. Dependency graph

```text
Exp6460 handoff ───────────────────────────────────────────────┐
Exp6461 source delta ─────────────────────────────────────────┤
                                                             │
Exp6462 raw canary ──▶ Exp6463 corpus v2 ──┬─▶ Exp6464 grounding
                                          │         │
                                          │         ▼
                                          │   Exp6465 objective ──▶ Exp6467 selection
                                          │
                                          └─▶ Exp6466 allocation

Exp6468 unique-event CSL ──▶ Exp6469 corruption/restart
          │                          │
          └──────────────┬───────────┘
                         ▼
                 Exp6470 independent audit

Exp6471 ARC safety ───────────────────────────────────────────┤
all terminal evidence ───────────────────────────────▶ Exp6472 capstone
```

Exp6470 is logically downstream but is not conductor-gated. It must record
missing or failed evidence. Exp6472 is also ungated.

## 10. Hardware requirements

| Experiments | Hardware | Requirement |
|---|---|---|
| Exp6462, Exp6463, Exp6468, Exp6469 | Two local RTX 3090 GPUs | Run the three mandated GGUF families through cached local files and embedded tokenizers. Record device samples and CPU fallback count. |
| Exp6460, Exp6461, Exp6464-Exp6467, Exp6470-Exp6472 | CPU and local disk | Replay exact checkers, reducers, source receipts, ARC traces, and audits. |
| All live generation | Local SSD | Use new raw-output paths, atomic rename, fsync where supported, byte count, SHA-256, and checkpoint manifests. |
| FPGA, TSU, NPU, Ising hardware | Not required | No board state changed. No unchanged probe or hardware performance claim is scheduled. |

The dual RTX 3090 pair is sufficient. No wishlist purchase blocks V556.

## 11. Promotion gates and stop rules

- A missing or zero-byte raw output is a failed event. It may not be replaced
  by another row under the same event ID.
- Each generation event has exactly one raw hash and one unit binding.
- A duplicated output may be reported, but it cannot count as a distinct
  learning or held event.
- Held partitions are sealed before inference. Any exposure retires the held
  claim for that unit.
- The exact checker runs before learning write admission and before release.
- A learned score, LLM score, energy, memory, or ARC heuristic is never a
  release oracle.
- Comparative tasks emit every unit row and pass the row-consistency lint.
- A blocked verdict names the failed check in `gate_check_summary`.
- If a changed rerun repeats its prior verdict, its task declares
  `retire_if_same_verdict: true`.
- Exp6471 cannot update the public ARC registry or claim a game or level solve.
- The capstone promotes only independently recomputed, attack-clean evidence.

## 12. Expected milestone decision

V556 can end in three honest states:

1. The raw substrate fails again. Retire this corpus path and do not run its
   gated causal chain.
2. The substrate passes but grounding, allocation, or energy does not improve
   held exact outcomes. Preserve the null and stop that mechanism.
3. Constraint routing or continuous learning earns an independently audited
   exact-outcome gain with complete unique-event provenance. Promote only that
   narrow claim.

The milestone is successful if it produces a trustworthy causal determination.
It does not need a positive result.

# Research Roadmap V565: External Constraint Transfer and Transactional Self-Learning

**Milestone:** `2026.08.565`  
**Sequence:** 565  
**Planning date:** 2026-08-23  
**Execution manifest:** `research-roadmap-next.yaml`  
**Status:** proposed  
**Experiment range:** Exp6527-Exp6540 (14 tasks, four phases)

## Purpose

V564 established two promising exact-guidance mechanisms on a bounded internal
procedural corpus: a Safety-Net branch router and refinement-valid conflict
memory. V565 asks the next harder question: do those mechanisms transfer to an
external, multi-turn constraint surface and remain safe when wired behind the
default-off production pipeline?

The milestone also removes two evidence bottlenecks before they can contaminate
that question. It corrects the still-flagged Exp6520 artifact, and it turns the
blocked ARC supervisor result into a live-path parser and outcome-receipt
reachability test rather than another zero-row downstream A/B.

Exact solvers and executable environments remain release authority. Learned
signals may rank, route, allocate work, or abstain. They may not prune the full
candidate set, certify their own outputs, or convert a development proxy into a
headline.

## What V564 proved

| Evidence | Terminal finding | V565 consequence |
|---|---|---|
| Exp6514 atomic shard transaction | Crash injection, resume, and atomic finalization passed on the local filesystem. | Reuse the transaction for external-corpus intake and row-producing experiments. Do not create another artifact framework. |
| Exp6518 and Exp6519 structural controls | The 18-unit pilot produced 126 matched branch rows across multiple families and seeds; independent replay certified charged held structural headroom with exact answer equality. | Test transfer to a content-pinned external corpus instead of expanding the same generators. |
| Exp6520 Safety-Net router | A compact router plus exception table and exact fallback beat the certified structural control on the bounded pilot, with zero held-table writes. The artifact still carries `flagged_adversarial=true`, `corrigendum_pending=true`, and an implausible `duration_s=0.080421`. | Correct evidence eligibility first. Replicate externally with held calibration and abstention before any broader claim. |
| Exp6521-Exp6523 conflict memory | Refinement-witness admission, rollback, restart, invalid-reuse veto, chronological held-future benefit, exact equality, and adaptive-validation savings all passed; the independent audit marked the continuous-learning claim eligible. | Move the mechanism behind a default-off `VerifyRepairPipeline` adapter and test an external chronological stream, retention, and future support. |
| Exp6524 ARC supervisor | Closed honestly as blocked because there were no outcome-bearing live supervisor receipts. | Repair the upstream tool-call and receipt reachability contract. Do not rerun the supervisor selection A/B yet. |
| Exp6525 GateMate continuity | No post-Exp6325 physical-state receipt existed; zero hardware commands ran. | Perform one final changed-state continuity/retirement decision. Never repeat `--detect` without a newer operator receipt. |
| Exp6526 capstone | Structural-router and continuous-learning evidence are row-supported, while ARC and GateMate remain blocked. | V565 can branch into external transfer and live-path reachability without claiming that V564 solved oracle-distinct verification or ARC. |

## The three largest gaps to the PRD vision

### Gap 1: the positive solver-guidance evidence is still internal and oracle-adjacent

The router and its labels were built from Carnot's own procedural exact
benchmark. That supports a solver-performance claim, not an oracle-distinct
verification moat. There is no independent external transfer result, no
family-blind external split, and no demonstration that abstention remains
calibrated under a different constraint grammar.

V565 uses a content-pinned slice of DRIFT-Bench (`arXiv:2605.23940`) across
seating, scheduling, and logic-grid streams. It regenerates every receipt
locally because the public repository warns that the original run databases
were corrupted. The milestone first measures non-learned headroom, then tests
the Safety-Net router, then independently audits shortcuts and calibration.

### Gap 2: continuous self-learning is not yet production-shaped

V564 proved exact conflict reuse on a chronological internal stream, but the
controller remains experiment-local. The PRD requires an autonomous
propose/verify/update loop with safety guards, persistence, and useful retained
learning. The open questions are whether memory can operate behind
`VerifyRepairPipeline`, whether query-local writes contaminate the same query,
whether earlier families are retained, and whether later exact-satisfying
support shrinks even when current cost improves.

V565 ships a default-off transactional adapter and evaluates scratch, frozen,
transactional-commit, and same-query-mutation arms on an external chronological
stream. It treats future support, retained-family performance, restart,
rollback, quarantine, and exact output equality as co-primary safety fields.

### Gap 3: the live ARC agent cannot yet produce evidence for its own supervisor

The scored vLLM path accepts tool-bearing requests only when auto tool choice
and a parser are configured. The pinned Qwen3.8-27B generator emits
`qwen3_xml`-shaped tool calls; the Hermes parser accepted requests but lifted
no calls. Exp6524 consequently found no valid outcome-bearing supervisor rows.
This is an upstream reachability defect, not evidence that trajectory
supervision is ineffective.

V565 tests parser registry/configuration, captured XML conformance, one bounded
live Qwen3.8 tool call, a mandated Qwen3.6 GGUF format control, and propagation
of a real outcome-bearing receipt through `E3AgentPolicy` / `make_carnot_agent`.
It makes no game-level solve claim and performs no Kaggle submission.

## Research findings incorporated

The dated source review is recorded in `research-references.md` under
`V565 planner refresh - 2026-08-23`.

- **DRIFT-Bench (`2605.23940`)** supplies the external multi-turn transfer
  surface and separates contradiction from residual satisfiable drift.
- **Memoir (`2607.20792`)** motivates read-only memory during a query and an
  exact-outcome commit boundary.
- **Verifier-Induced Support Reshaping (`2608.00220`)** makes future
  exact-satisfying support a required continual-learning metric.
- **Distributional EBMs (`2605.18871`)** motivates decomposed analytical and
  learned routing, held calibration, uncertainty, and abstention, while its
  code confound motivates identity-shortcut attacks.
- **Solver-Hard Is Not Model-Hard (`2607.17047`)** requires separate structural
  hardness and surface-realization strata for SOTA-model diagnostics.
- **DC energy reasoning** and **hard linear decision-rule networks** remain
  future optimizer/architecture controls; neither justifies a new answer-level
  generation lineage in V565.
- Extropic's Z1 remains taped out with 2027 early access; Kona remains
  proprietary. Neither is an executable dependency.

## Scientific invariants

1. **External means content-pinned.** Record upstream URL, revision, license,
   file hashes, local transformation hashes, and exact replay receipts.
2. **Rows before aggregates.** Every comparison emits one row per turn,
   instance, model, seed, arm, or condition. Aggregates are recomputed from
   those rows.
3. **Exact authority is separate.** Exact solvers validate assignments and
   conflicts. A learned router cannot certify an answer or write an exception
   based on held outcomes.
4. **No candidate deletion.** Structural or learned guidance may reorder a
   complete candidate set. Native exact fallback remains reachable.
5. **Transactional learning.** Memory is frozen within a query. A write is
   admitted only after exact outcome validation and a refinement witness.
6. **Future support is co-primary.** Current cost reduction cannot hide
   reduced retained-family performance or future exact-satisfying support.
7. **No retired answer transport.** V565 does not retry finite-ID, grammar,
   stop-token, or parser-only generated benchmark answers. The SOTA GGUF task
   uses paired embeddings only. The ARC parser task is a live tool-call
   reachability test under the standing AVO directive.
8. **ARC live path only.** ARC credit requires reachability from
   `E3AgentPolicy` / `make_carnot_agent`. V565 claims no level or game solve.
9. **No unchanged hardware probe.** GateMate receives zero commands without a
   new dated physical-state receipt. KV260 and PolarFire remain terminal.
10. **Closed verdict classes.** Every artifact includes `verdict_class` from
    `positive | circular_positive | null | blocked | disqualified | partial`.

## Model policy

Two tasks need model inference.

- **Exp6532** calls `cached_sota_pair(gpu_indices=(0, 1))`, which resolves all
  three mandated GGUFs:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. It collects paired embeddings only and
  makes no generated-answer correctness claim.
- **Exp6538** keeps `unsloth/Qwen3.8-27B-GGUF` as the live ARC generator and
  includes `unsloth/Qwen3.6-35B-A3B-GGUF` as the mandated local format/control
  model. Qwen3.6 cannot substitute for Qwen3.8 in the live-agent claim.

If the mandated cached models are unavailable, the affected task closes as
blocked with `gate_check_summary`. Qwen3.5-0.8B or gemma-4-E4B-it may run only a
CPU smoke test and can never supply a headline row.

Per the repository's standing Codex-default rule, all tasks declare
`agent_type: codex` and `model: gpt-5.5`.

## Architecture after V565

```text
                     V564 immutable evidence
                 +-----------------------------+
                 | structural rows | conflicts |
                 +----------+------------+-----+
                            |            |
                  Exp6527 eligibility    |
                       corrigendum        |
                            |            |
External DRIFT-Bench        |            |
URL/revision/license        |            |
        |                   |            |
  Exp6529 intake ------> Exp6530 audit   |
        |                   |            |
        +--------------+----+            |
                       |                 |
             Exp6531 external       Exp6535 default-off
             structural headroom    conflict-memory adapter
                       |                 |
               Exp6533 calibrated       |
               Safety-Net transfer      |
                       |            Exp6536 chronological
               Exp6534 independent  external self-learning
               shortcut/calibration     |
               audit                Exp6537 retention,
                                        support, rollback audit

  Exp6532 mandated-SOTA paired-embedding surface diagnostic
      (diagnostic only; never validity authority)

  Captured Qwen3.8 XML + live generator + Qwen3.6 control
                       |
             Exp6538 qwen3_xml/tool-call reachability
                       |
        live E3 outcome-bearing supervisor receipts
              (future supervisor A/B; no solve here)

  Exp6539 GateMate receipt gate ----> zero commands or one bounded action

  All terminal artifacts ----------> Exp6540 independent capstone
```

## Phase 0 - Evidence and external-corpus contracts

### Exp6527 - V565 activation manifest and V564 evidence eligibility corrigendum

**Question:** Can V565 establish a non-retired immutable root while correcting
the unresolved Exp6520 adversarial/duration receipt?

Recompute Exp6518-Exp6523 claims from rows, rerun the live adversarial verifier,
measure a real nontrivial execution interval for Exp6520 validation, and
separate historical artifact fields from the corrected eligibility record. Do
not edit historical artifacts. The root opens only if the router and CSL inputs
are content-addressed, row-consistent, and not dependent on retired task IDs.

**Deliverable:**
`results/experiment_6527_v565_evidence_eligibility_corrigendum.json`

### Exp6528 - Execution-time V565 source, model, and method contract

**Question:** Did any post-planning source, model-cache state, or product update
change the method before outcomes are visible?

Perform a focused low-concurrency source delta, verify DRIFT-Bench provenance,
refresh Semantic Scholar counts without inventing rate-limited data, and freeze
the external split, abstention, transactional-learning, support, ARC parser,
and hardware stop rules.

**Deliverable:**
`results/experiment_6528_v565_source_model_method_contract.json`

### Exp6529 - Content-pinned DRIFT-Bench intake and exact replay commitment

**Question:** Can Carnot produce a local, immutable, solver-replayed external
constraint slice without inheriting corrupt upstream run receipts?

Pin revision and license, select a bounded balanced slice from seating,
scheduling, and logic-grid streams, preserve chronological turns, regenerate
Z3 labels, and seal train/development/held family-blind splits. Use the V564
atomic transaction. The task produces both a terminal artifact and a hashed
JSONL fixture.

**Deliverables:**
`results/experiment_6529_drift_bench_external_intake.json` and
`results/fixtures/v565_drift_bench_external_slice.jsonl`

### Exp6530 - Independent external-corpus and split audit

**Question:** Are the external rows, exact labels, chronology, splits, and
provenance independently reproducible?

Always run. Re-download or independently inspect the pinned source, replay a
precommitted sample and every held row, recompute hashes and aggregates, and
attack duplicate turns, family aliases, row ordering, answer leakage, solver
version drift, and omitted hard rows.

**Deliverable:**
`results/experiment_6530_external_constraint_corpus_audit.json`

## Phase 1 - External transfer and surface diagnostics

### Exp6531 - External structural-guidance headroom replication

**Gate:** `Exp6530.external_constraint_corpus_audited_ready_score == 1.0`.

**Question:** Do the V564 non-learned structural controls retain charged held
benefit on the external multi-turn corpus?

Compare native dynamic ordering, random ordering, analytical structural order,
bounded refocus, and one-shot critical-variable enumeration under matched
budgets. Report conflicts, decisions, propagations, wall time, censoring,
answer equality, per-family effects, and chronology position. A single family
or one uncensored win cannot open the router gate.

**Deliverable:**
`results/experiment_6531_external_structural_headroom_replication.json`

### Exp6532 - Mandated-SOTA paired-embedding surface-sensitivity audit

**Gates:** external corpus audited and V565 method contract ready.

**Question:** How much do proof-preserving surface realizations move local SOTA
representations independently of exact structural hardness?

For all three mandated GGUFs, collect matching-base paired embeddings for
canonical, entity-relabeled, clause-reordered, and paraphrased versions of the
same exact rows. Stratify by solver hardness and family. Measure paired
distance, neighbor stability, model/family identification shortcuts, cache and
runtime receipts, and repeatability. This is a diagnostic; it does not score
answers or train a verifier.

**Deliverable:**
`results/experiment_6532_sota_paired_embedding_surface_audit.json`

### Exp6533 - Calibrated Safety-Net router external transfer

**Gates:** V565 evidence root ready, external corpus audited, and external
structural headroom candidate score equals one.

**Question:** Can the V564 compact router improve charged external search cost
while an exception table, uncertainty abstention, and native exact fallback
preserve completeness?

Freeze features and calibration before held outcomes. Compare native,
best non-learned structural, router without abstention, and calibrated
Safety-Net arms. No held table writes; no candidate pruning. Require exact
answer equality, bounded storage, live influence, multi-family support, and
positive charged benefit beyond the best structural control.

**Deliverable:**
`results/experiment_6533_external_calibrated_safety_net_router.json`

### Exp6534 - Independent external-router shortcut and calibration audit

**Question:** Does the external router survive independent row reduction,
calibration replay, family/model-identity attacks, feature permutation, and
duration/cost verification?

Always run and close the eligibility field. Recompute all comparison metrics
from rows, replay the exact fallback, inspect feature provenance, verify that
no held writes occurred, and distinguish solver-performance evidence from an
oracle-distinct verification claim.

**Deliverable:**
`results/experiment_6534_external_router_independent_audit.json`

## Phase 2 - Production-shaped continuous self-learning

### Exp6535 - Default-off transactional conflict-memory pipeline adapter

**Gate:** V565 evidence root ready.

**Question:** Can the V564 exact conflict-memory controller be reached safely
from `VerifyRepairPipeline` while leaving the default path byte-for-byte and
behaviorally unchanged?

Implement the smallest adapter using the existing factor-cache shadow pattern.
Memory is read-only during a query and commits after exact validation. Require
versioned refinement witnesses, content-addressed persistence, restart,
rollback, quarantine, bounded eviction, concurrent-writer refusal, and an
unchanged default-off path.

**Deliverable:**
`results/experiment_6535_verify_repair_conflict_memory_adapter.json`

### Exp6536 - Prospective external chronological conflict self-learning A/B

**Gates:** default-off adapter ready and external corpus audited.

**Question:** Does transactional conflict memory reduce charged future work on
the external multi-turn stream beyond scratch and frozen controls without
shrinking future exact-satisfying support?

Compare scratch, frozen memory, transactional post-query commit, and a
same-query-mutation contamination arm under matched admitted-event dose.
Evaluate chronology once, with no rewind after held outcomes. Report current
cost, held-future cost, prefix retention, future support, exact equality,
unsafe writes/uses, restart, rollback, quarantine, and cross-family transfer.

**Deliverable:**
`results/experiment_6536_external_chronological_conflict_self_learning.json`

### Exp6537 - Independent retention, support, and rollback audit

**Gate:** `Exp6536.prospective_csl_execution_complete_score == 1.0`.

**Question:** Is the external continuous-learning claim still eligible after
independent chronological replay and adversarial state attacks?

Recompute all rows, restore checkpoints in a fresh process, replay admitted and
rejected conflicts, inject invalid refinement witnesses and torn writes, audit
matched update dose, compare future support, and require no retained-family
regression beyond the preregistered tolerance.

**Deliverable:**
`results/experiment_6537_external_csl_retention_support_audit.json`

## Phase 3 - Live-path and hardware continuity, then synthesis

### Exp6538 - ARC qwen3_xml tool-call and supervisor-receipt reachability

**Question:** Can the live E3 path lift one real tool call from the pinned local
generator and propagate an outcome-bearing supervisor receipt?

First test the pinned vLLM parser registry and captured Qwen3.8 XML. Then run a
bounded local live smoke with auto tool choice plus `qwen3_xml`, and a mandated
Qwen3.6 GGUF format control. Route the lifted call through
`E3AgentPolicy` / `make_carnot_agent`, record the executed outcome, and verify
that `arm_outcomes` reaches the scored-harness row. Stop after the receipt
contract is demonstrated or a named precondition fails. No public re-solve,
game-level solve, Kaggle submission, source read, or offline BFS is allowed.

**Deliverable:**
`results/experiment_6538_arc_qwen3_xml_receipt_reachability.json`

### Exp6539 - GateMate changed-state continuity and retirement decision

**Question:** Is there a new operator-authored physical-state receipt after
Exp6525 that authorizes one bounded GateMate action?

If not, run zero hardware commands, close the milestone slot honestly, and
retire the unchanged probe scope mechanically if the same verdict recurs. If a
new receipt exists, validate exact board/cable/port/power/DirtyJTAG fields and
perform at most one predeclared detection action before stopping.

**Deliverable:**
`results/experiment_6539_gatemate_changed_state_retirement.json`

### Exp6540 - Independent V565 capstone and next-state decision

**Question:** Which V565 claims survive independent row-first synthesis, and
what is the smallest justified next milestone?

Always run. Recompute every gate from artifacts and rows; preserve blocked,
null, circular, partial, and disqualified outcomes. Separate internal solver
performance, external transfer, production-shaped CSL, model diagnostics, ARC
reachability, and hardware status. Recommend expansion only where breadth,
charged benefit, exact equality, and independent eligibility all hold.

**Deliverable:**
`results/experiment_6540_v565_independent_capstone.json`

## Dependency graph

```text
Exp6527 evidence root ------------------------+----------------------+
                                              |                      |
Exp6528 method contract ----+                 |                      |
                            |                 |                      |
Exp6529 external intake --> Exp6530 audit ----+--> Exp6531 headroom  |
                            |                 |         |            |
                            +--> Exp6532 SOTA |         v            |
                                 embedding    +--> Exp6533 router    |
                                                      |              |
                                                      v              |
                                                Exp6534 audit        |
                                                                     |
Exp6521 immutable conflict mechanism + Exp6527 --> Exp6535 adapter --+
Exp6530 -------------------------------------------> Exp6536 CSL
Exp6535 -------------------------------------------> Exp6536 CSL
Exp6536 execution complete ------------------------> Exp6537 audit

Exp6538 ARC reachability       (independent live-path slot)
Exp6539 GateMate continuity    (independent receipt-gated slot)

Exp6527-Exp6539 terminal artifacts ----------------> Exp6540 capstone
```

No task has a `requires:` or `gated_on:` edge to a retired experiment ID.
Historical artifacts are read directly by path and hash only.

## Hardware requirements

| Resource | Tasks | Requirement and stop rule |
|---|---|---|
| CPU, RAM, local disk | Exp6527-Exp6531, Exp6533-Exp6537, Exp6540 | Exact solver replay, JSONL fixtures, transactional checkpoints, and audits. Preflight disk before external intake. Charge all preprocessing and exact fallback costs. |
| Dual RTX 3090 | Exp6532 | Resolve the three mandated GGUFs through `cached_sota_pair(gpu_indices=(0, 1))`; run models sequentially where residency conflicts. Record cache hashes, GPU assignment, VRAM, thermals, wall time, and failures. Missing mandated cache blocks the scientific task. |
| Local vLLM/llama.cpp and dual RTX 3090 | Exp6538 | Pinned Qwen3.8-27B is the live ARC generator; Qwen3.6-35B-A3B is the mandated format control. Use bounded prompts and one reachability smoke. No Kaggle quota or submission dependency. |
| GateMate A1 board | Exp6539 | No command without a dated operator receipt newer than Exp6525. At most one predeclared action after a valid receipt. No unchanged `--detect` retry. |
| KV260 and PolarFire | none | Terminal according to the hardware ledger; no repeated task. |
| Extropic XTR-0/Z1 | none | No authenticated route. Z1 early access remains 2027; no latency, power, or execution claim. |
| Ryzen AI/XDNA NPU | none | Deferred after repeated unchanged precondition failures; no reinstall probe. |

## Gates and stop rules

- Exp6531 cannot run without an independently audited external corpus.
- Exp6533 cannot run unless external non-learned structural headroom is
  positive and the Exp6520 evidence root is eligible.
- Exp6536 cannot run unless both the external corpus and the default-off
  pipeline adapter are ready.
- Exp6537 cannot run unless Exp6536 writes
  `prospective_csl_execution_complete_score=1.0` with that exact spelling.
- A failed structured gate writes `verdict_class=blocked` and a populated
  `gate_check_summary` naming the field and observed value.
- A positive comparative claim requires per-unit rows, exact answer equality,
  charged wall time, breadth beyond one family, and no contradicted aggregate.
- Exp6532 cannot fall back to legacy tiny models for a headline result.
- Exp6538 stops after one demonstrated live receipt or a named substrate block;
  it does not continue into a supervisor-policy A/B or game solve.
- Exp6539 executes zero hardware commands without a new receipt and carries
  `retire_if_same_verdict: true` for the repeated blocked scope.
- Exp6540 never upgrades a circular exact-oracle result into an
  oracle-distinct verification claim.

## Deliverable index

| Exp | Primary deliverable |
|---:|---|
| 6527 | `results/experiment_6527_v565_evidence_eligibility_corrigendum.json` |
| 6528 | `results/experiment_6528_v565_source_model_method_contract.json` |
| 6529 | `results/experiment_6529_drift_bench_external_intake.json` |
| 6530 | `results/experiment_6530_external_constraint_corpus_audit.json` |
| 6531 | `results/experiment_6531_external_structural_headroom_replication.json` |
| 6532 | `results/experiment_6532_sota_paired_embedding_surface_audit.json` |
| 6533 | `results/experiment_6533_external_calibrated_safety_net_router.json` |
| 6534 | `results/experiment_6534_external_router_independent_audit.json` |
| 6535 | `results/experiment_6535_verify_repair_conflict_memory_adapter.json` |
| 6536 | `results/experiment_6536_external_chronological_conflict_self_learning.json` |
| 6537 | `results/experiment_6537_external_csl_retention_support_audit.json` |
| 6538 | `results/experiment_6538_arc_qwen3_xml_receipt_reachability.json` |
| 6539 | `results/experiment_6539_gatemate_changed_state_retirement.json` |
| 6540 | `results/experiment_6540_v565_independent_capstone.json` |


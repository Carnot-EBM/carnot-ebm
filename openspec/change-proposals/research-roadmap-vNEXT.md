# Carnot Research Roadmap V602: Exact Relational Facts, Structural Self-Learning, and Live Tool Credit

**Milestone:** `2026.09.602`  
**Status:** Planned  
**Task contract:** 11 tasks, `exp6874` through `exp6884`, in conductor order  
**North star:** reduce hallucinations with local SOTA models, exact constraints, and safe learning  
**ARC floor:** one adapter-disabled live generalization branch; no game-level solve claim  
**Self-learning floor:** one prospective update-versus-read-only comparison

## Executive Decision

V601 closed two mechanisms. Fixed-sequence semantic likelihood did not survive nuisance controls.
The bounded reliability update did not beat read-only memory. V602 does not version either method.

The new semantic unit is an Enoki-style text-anchored relational fact. A local model may propose a
fact. Carnot maps the fact to a bounded ASP atom and checks it with the qualified Exp6274 compiler
and an independent exact solver. The learned path never owns release authority.

The new learning unit is structural routing over exact-admitted relation and constraint records.
The decisive control is read-only retrieval. The update arm earns only a circular-positive class
because the exact verifier is also the update and evaluation authority.

The live ARC branch restores work that appeared in the V601 design document but was absent from
the executable YAML. It first measures shared-context headroom. It then collects first-party tool
gaps and compares delivery with withholding on the same pre-action opportunity.

## What V601 Proved

| Evidence | Result | V602 consequence |
|---|---|---|
| Exp6865 evidence contract | The method-change contract was complete. The conductor quarantined its short deterministic run. | Recompute the evidence disposition under one explicit no-LLM substrate contract. |
| Exp6866 tokenizer binding | All three canonical tokenizer bindings passed. | Reuse the bindings. Do not reopen tokenizer provenance. |
| Exp6867 preregistration | The semantic split and nuisance controls were ready. | Treat its split as frozen evidence. |
| Exp6868 scoring | All 880 planned cells completed on the three required GGUF families. | Reuse the raw rows. Do not rerun fixed-sequence likelihood. |
| Exp6869 calibration | Effect, replication, and nuisance gates failed. | Close scalar semantic compatibility as a claim mechanism. |
| Exp6870 sealed audit | The structured gate blocked because Exp6869 was null. | Preserve the block as a correct terminal result. |
| Exp6871 opportunity stream | 765 exact chronological events were available. | Reuse the chronology patterns, not the reliability feature set. |
| Exp6872 controller | The controller fired several actions and admitted no harmful transition. | Safety was necessary but not sufficient. |
| Exp6873 sealed audit | Quarantine beat frozen in every order but never beat read-only. | Retire reliability-state updates unless the learned object changes. |
| V601 ARC phase | Four tasks existed in the design document but not in the nine-task YAML. They did not run. | Restore the work with current task IDs, fields, and gates. |

V601 also exposed an evidence-tooling problem. Several short CPU artifacts were stamped as
`DURATION_TOO_SHORT` because their prose substrate names looked like live work. V602 begins with an
append-only requalification. It does not rewrite V601 artifacts or weaken live-model duration rules.

## Three Largest Gaps to the PRD Vision

### Gap 1: No useful local-model semantic constraint interface

Carnot has exact solvers and a bounded ASP energy compiler. It does not yet have a robust path from
all three required GGUF families to source-grounded semantic atoms. V601 showed that scalar token
likelihood is not that path.

### Gap 2: Continuous learning has no value beyond read-only memory

The V601 updater was safe but unnecessary. FR11 requires useful prospective adaptation, retention,
rollback, and held-out evaluation. V602 changes the learned object from source reliability weights
to small typed relation and constraint-routing records.

### Gap 3: Live ARC intervention credit is still missing

The live seam and first-party receipt schema exist. Authentic adapter-disabled tool-gap chains and
matched outcome effects do not. Shared-context truncation can still confound any tool comparison.

## External Research Incorporated

- **Enoki** (`2609.00581`) supplies the shared text-anchored relation record. V602 uses it for
  source spans, evidence spans, relation tuples, and exact atom linkage.
- **ASP Energised** (`2607.08136`) motivates declarative semantics plus energy factors. V602 uses
  only the bounded local compiler and exact stable-model parity already qualified by Exp6274.
- **When Continual Learning Moves to Memory** (`2604.27003`) supports abstract typed records and a
  strong read-only control. V602 does not store raw trajectories as learned truth.
- **Repair, Not Improvement** (`2608.13959`) requires separate format, tool-needed, abstain, and
  exact-outcome measures in the ARC branch.
- **The Hallucination Signal Is a Mean Shift** (`2608.28930`) is retained as a future white-box
  lead. V602 does not reopen hidden-state scoring without a GGUF multi-layer extraction contract.
- **Learned multiscale sampling** (`2608.31114`) remains a sampler lead. It does not close a V602
  blocker and does not justify a hardware task.

The full dated source record is in `research-references.md`, V602 Planner Refresh.

## Target Architecture

```text
                         evidence contract (Exp6874)
                                     |
                                     v
  local SOTA GGUFs --> anchored relation tuple --> bounded ASP atom
       Exp6876               Exp6875/6876             Exp6274
          |                         |                     |
          |                         +------ exact --------+
          |                                      authority
          v
  frozen raw relation rows --> independent held audit (Exp6877)
                                     |
                                     v
                           chronological events (Exp6878)
                                     |
                  +------------------+------------------+
                  |                  |                  |
             no memory          read-only         bounded update
                  |                  |                  |
                  +---------- exact later outcomes ----+
                                     |
                          sealed CSL audit (Exp6880)

  live E3AgentPolicy --> context/headroom qualifier (Exp6881)
                                     |
                         first-party tool gaps (Exp6882)
                                     |
                    matched deliver / withhold (Exp6883)
                                     |
                             exact next outcomes

                    cold branch synthesis (Exp6884)
```

The semantic and ARC branches are independent. The capstone reads both. No science task depends on
a positive result from the other branch.

## Model and Runtime Contract

Every task that performs new LLM inference must use `cached_sota_pair()` and native embedded GGUF
tokenizers. It must include at least one required model in `MODEL_SPECS`. The main relation corpus
uses all three:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models are allowed only for CPU smoke tests. They cannot support a headline result.
GGUF repositories must not be passed to `AutoTokenizer.from_pretrained()`.

## Phase A: Evidence and Exact Relational Facts

### Exp6874: V602 evidence, substrate, and manifest-parity contract

Recompute V601 terminal evidence from primary artifacts and conductor rows. Record stored and fresh
adversarial dispositions. Qualify short CPU results only with explicit no-LLM code and command
receipts. Detect the V601 document/YAML task mismatch. Freeze the 11-task V602 manifest before any
science task. This is infrastructure slot one.

**Deliverable:** `results/experiment_6874_v602_evidence_substrate_manifest_contract.json`  
**Gate output:** `v602_evidence_contract_ready_score`

### Exp6875: Text-anchored relation and ASP atom fixture

Define a bounded relation record with source span, subject, predicate, object, evidence span,
normalization, and ASP atom. Build calibration and sealed-held fixtures over qualified Exp6274
families. Check relation-to-atom parity with the bounded compiler and an independent solver. This is
infrastructure slot two.

**Deliverable:** `results/experiment_6875_text_anchored_relation_asp_fixture.json`  
**Gate output:** `relation_fact_fixture_ready_score`

### Exp6876: Three-family relational-fact extraction corpus

Run all three required GGUF families on the frozen natural-language fixtures. Preserve every raw
output, parse failure, abstention, span, tuple, and atom candidate. Do not use constrained-schema
decoding, a model judge, or an external text scorer. Readiness means complete authentic acquisition,
not model quality.

**Deliverable:** `results/experiment_6876_three_family_relational_fact_corpus.json`  
**Gate outputs:** `relational_fact_corpus_ready_score`, `model_family_coverage_ready_score`

### Exp6877: Independent held relational-fact qualification

Open the held sidecar once in a fresh reducer. Recompute span grounding, tuple accuracy, parse
coverage, ASP compilation, and exact semantic validity for every model and family. Preserve null and
failed cells. The exact verifier is oracle authority, so any passing class is
`circular_positive`, never `positive`.

**Deliverable:** `results/experiment_6877_independent_relational_fact_qualification.json`  
**Gate outputs:** `relational_fact_qualification_complete_score`, `eligible_learning_event_count`

## Phase B: Structural Continuous Self-Learning

### Exp6878: Exact relational-routing opportunity stream

Convert eligible Exp6877 rows into a chronological stream with later exact outcomes. Freeze no
memory, read-only, bounded-update, quarantine, and matched-placebo actions. Include held families,
poison, delayed correction, conflict, restart, and rollback rows. No update occurs in this task.

**Deliverable:** `results/experiment_6878_exact_relational_routing_stream.json`  
**Gate output:** `relational_routing_stream_ready_score`

### Exp6879: Prospective relational constraint-routing A/B

Reuse the compositional routing mechanism that was positive on the exact Exp6790 fixture. Apply it
prospectively to the live-model relation stream. Update only between events after exact outcomes.
Compare no memory, read-only, bounded update, quarantine, and placebo on identical rows and orders.
This is the milestone's required continuous self-learning experiment.

**Deliverable:** `results/experiment_6879_prospective_relational_constraint_routing_ab.json`  
**Gate output:** `relational_routing_comparison_complete_score`

### Exp6880: Sealed structural self-learning audit

Cold-replay at least five frozen event orders. The update arm must beat read-only and no memory on
held future rows. It must also pass retention, poison, delay, persistence, restart, and rollback
checks. If it repeats V601's no-gain-over-read-only verdict, retire this live structural-routing
attempt.

**Deliverable:** `results/experiment_6880_sealed_structural_self_learning_audit.json`

## Phase C: Context-Safe Live ARC Tool Credit

### Exp6881: Shared-context headroom and receipt qualification

Instrument the canonical `arc_scored_path_lever_harness` path. Record actual slot count, actual
`n_ctx`, prompt tokens, requested tokens, generated reasoning, final-channel tokens, truncation
location, VRAM, and teardown for each attempt. Revalidate the Exp6859 first-party receipt schema.
Do not solve a game or read game source. This task blocks max-token-only repairs.

**Deliverable:** `results/experiment_6881_arc_context_headroom_receipt_qualification.json`  
**Gate outputs:** `arc_context_headroom_ready_score`, `tool_gap_receipt_contract_ready_score`

### Exp6882: Adapter-disabled first-party tool-gap accrual

Run the live `E3AgentPolicy` path with adapters disabled and the qualified context contract. Keep
first-party selfparse tool demands, refused calls, actions, next states, and exact outcomes. Require
replayable pre-action alternatives. Do not read game source or claim a solve.

**Deliverable:** `results/experiment_6882_adapter_disabled_tool_gap_accrual.json`  
**Gate outputs:** `tool_gap_opportunity_ready_score`, `replayable_tool_gap_opportunity_count`

### Exp6883: Matched tool delivery versus withholding

For each replayable opportunity, branch from the same pre-action state. Deliver the demanded tool
in one arm and withhold it in the other. Match model, prompt, context, generation, action, and stop
budgets. Report tool-needed and abstain rows separately. Credit only exact next outcomes. Any level
advance remains live-agent provenance and is not a milestone solve claim.

**Deliverable:** `results/experiment_6883_matched_tool_delivery_withholding_ab.json`

## Phase D: Cold Synthesis

### Exp6884: Independent V602 capstone

Recompute every branch disposition from rows and primary artifacts. Recheck gates, prior-failure
retirement, model receipts, exact authority, ARC provenance, and document/YAML parity. Do not rerun
science. A null, block, or disqualification is terminal evidence.

**Deliverable:** `results/experiment_6884_v602_independent_capstone.json`

## Dependency Graph

```text
Exp6874 --> Exp6875 --> Exp6876 --> Exp6877 --> Exp6878 --> Exp6879 --> Exp6880
    |
    +------> Exp6881 --> Exp6882 --> Exp6883

Exp6884 reads Exp6874..Exp6883 but is ungated.
```

| Downstream task | Upstream field | Condition |
|---|---|---|
| Exp6875 | `exp6874.v602_evidence_contract_ready_score` | `== 1` |
| Exp6876 | `exp6875.relation_fact_fixture_ready_score` | `== 1` |
| Exp6877 | `exp6876.relational_fact_corpus_ready_score` | `== 1` |
| Exp6877 | `exp6876.model_family_coverage_ready_score` | `== 1` |
| Exp6878 | `exp6877.relational_fact_qualification_complete_score` | `== 1` |
| Exp6878 | `exp6877.eligible_learning_event_count` | `>= 90` |
| Exp6879 | `exp6878.relational_routing_stream_ready_score` | `== 1` |
| Exp6880 | `exp6879.relational_routing_comparison_complete_score` | `== 1` |
| Exp6881 | `exp6874.v602_evidence_contract_ready_score` | `== 1` |
| Exp6882 | `exp6881.arc_context_headroom_ready_score` | `== 1` |
| Exp6882 | `exp6881.tool_gap_receipt_contract_ready_score` | `== 1` |
| Exp6883 | `exp6882.tool_gap_opportunity_ready_score` | `== 1` |
| Exp6883 | `exp6882.replayable_tool_gap_opportunity_count` | `>= 12` |

All structured gates name fields that the upstream task must emit at top level. Every upstream task
is in this roadmap. The capstone remains ungated so blocked branches still receive a final record.

## Failed-Scope and Retirement Boundaries

- Exp5923 retired schema-supported ConstraintIR reprompting after zero exact semantic success.
  V602 uses text-anchored relations and a prequalified ASP atom map. It does not use schema-guided
  decoding. Exp6876 and Exp6877 carry the prior failure mechanically.
- Exp5909 found no exact synthesis gain from structured prompt arms. Exp6877 tests grounded fact
  extraction and exact qualification, not prompt-based repair.
- Exp6678 failed to make an exact constraint-family stream ready. Exp6878 has a new upstream: held
  exact-admitted relation rows from all three required model families.
- Exp6873 found no learning gain over read-only. Exp6879 and Exp6880 change the learned object to
  compositional relation routing. A repeated no-gain verdict retires this live attempt.
- Exp6776 blocked on exclusive GPU availability. Exp6881 requires task-owned GPU and server
  receipts before any live ARC row.
- Exp6845 had zero first-party tool-gap obligations. Exp6882 uses the shipped Exp6859 receipt seam
  and context-qualified selfparse path.
- Exp6682's matched supervisor A/B ended with verification failure. Exp6883 uses tool delivery,
  exact pre-action replay identity, and task-owned focused verification.

No task reuses a retired experiment ID. No current dependency points to a retired task.

## Hardware Requirements

| Resource | Use | Claim boundary |
|---|---|---|
| Two RTX 3090 GPUs | Exp6876 three-family corpus and Exp6881-6883 live ARC work | Record per-process GPU UUID, placement, offload, VRAM, lease, and teardown. |
| Local CPU and RAM | ASP enumeration, relation reduction, chronological routing, audits | Use explicit no-LLM or deterministic substrates. Do not inherit live duration floors. |
| KV260, GateMate, PolarFire | Not in the blocking graph | Existing terminal receipts stand. No speed or availability claim. |
| Extropic TSU/Z1 | Not available locally | No hardware execution, latency, power, or throughput claim. |

If a required GPU lease is unavailable, the task writes a terminal blocked artifact. It must not
fall back to a legacy small model for a headline cell.

## Experimental Validity and Claim Rules

- Add the task's `REQ-*` and scenarios before implementation code.
- State `inference_substrate` and model identity in every terminal artifact.
- Put a principle next to every required artifact field through `field_principles`.
- Preserve one per-unit row for every model, fixture, event, arm, order, game, and condition used in
  a comparison.
- A blocked verdict records `gate_check_summary` with the failed check and observed value.
- Every task emits the closed `verdict_class` enum.
- `verifier_is_oracle=true` forbids `verdict_class=positive`. Use `circular_positive` when all
  declared checks pass under exact oracle authority.
- The ARC live path is `E3AgentPolicy` through `arc_scored_path_lever_harness`, with adapters off.
- ARC tasks do not read game source, use offline BFS, use a per-game adapter, or claim a solve.
- Any incidental ARC level advance records `solve_provenance=live_agent_self_discovery` and remains
  uncredited as a milestone solve.
- Do not reopen generated-text external energy scoring, fixed-sequence semantic likelihood,
  schema-supported ConstraintIR reprompting, or the V601 reliability updater.
- Do not modify `scripts/research_conductor.py`.

## Milestone Exit Criteria

V602 succeeds as a research milestone when all 11 tasks reach terminal artifacts and the capstone
reconciles their rows. Scientific outcomes may be positive, circular-positive, null, blocked,
partial, or disqualified.

A relational-fact claim requires authentic three-family model receipts, sealed held reduction,
span grounding, and exact ASP parity. A self-learning claim requires the update arm to beat
read-only across the replication rule with zero safety failures. An ARC tool claim requires
context-safe replayable opportunities and matched exact outcomes. No aggregate may override its
own per-unit rows.

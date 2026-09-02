# Carnot Research Roadmap V603: Exact Relations, Prospective Memory, and Live Tool Margins

**Milestone:** `2026.09.603`  
**Status:** Planned  
**Task contract:** 13 tasks, `exp6885` through `exp6897`, in conductor order  
**North star:** reduce hallucinations with local SOTA proposals, exact constraints, and safe learning  
**ARC floor:** one adapter-disabled live generalization branch; no game-level solve claim  
**Self-learning floor:** one prospective update-versus-read-only comparison

## Executive Decision

V602 proved a process fact, not a scientific result. Its design document promised 11 tasks, but its
executable YAML contained four. Exp6874 detected that mismatch. Exp6875 then stopped at the failed
root gate, and Exp6876 and Exp6877 ended as retired-upstream skips. The relation, learning, and ARC
methods did not run.

V603 restores the unfinished science under a different execution shape. The manifest audit is
independent and advisory. The exact-relation branch and live ARC branch have separate roots. The
blind-spot audit and capstone are ungated. One failed contract cannot erase the other branch or the
terminal synthesis.

The semantic interface is a text-anchored relation record. Enoki, a rule arm, and the required
local GGUFs may propose records. A bounded ASP map and an independent exact solver own admission.
The learning interface is a prospective typed-relation memory with no-memory, read-only lexical,
read-only structural, and bounded-update controls. The live ARC interface first repairs the
shared-context trace, then tests tool delivery with a CoBRA-style paired outcome margin.

## What V602 Proved

| Evidence | Result | V603 consequence |
|---|---|---|
| Exp6874 manifest contract | The V602 document listed 11 tasks and `exp6874`-`exp6884`; the activated YAML listed four and ended at `exp6877`. | Validate the exact 13-task V603 contract, but do not gate science on the audit. |
| Exp6875 relation fixture | The conductor wrote `blocked_gate_check_failed` because Exp6874 readiness was zero. | Re-run the narrowed fixture as an independent branch root and carry the prior block. |
| Exp6876 relation corpus | It never reached an agent or artifact because its upstream was retired. | Use a new ID and the same material method change: anchored relations, public Enoki assets, and all three required GGUFs. |
| Exp6877 held qualification | It never reached an agent or artifact because its upstream was retired. | Restore a sealed reducer behind the new corpus contract. |
| Scientific evidence | No V602 relation, self-learning, ARC, or hardware claim was produced. | Do not describe the unfinished V602 methods as null or positive. |

The last usable scientific evidence remains V601 and earlier. Exp6869 rejected scalar semantic
compatibility after effect, family, and nuisance gates failed. Exp6873 found no prospective update
benefit over read-only memory. Exp6844 and Exp6845 found no outcome-credit headroom or first-party
tool-gap support. Exp6859 proved only the tool receipt contract. The new work changes the learned
object, uses independent exact authority, and repairs the live context path before a tool A/B.

## Three Largest Gaps to the PRD Vision

### Gap 1: No qualified semantic path from local models to exact constraints

Carnot has exact executors and a bounded ASP energy compiler. It does not have a held-qualified path
from source text and the three required GGUF families to source-grounded semantic atoms. Scalar
likelihood failed, schema-supported ConstraintIR is retired, and V602's relation work did not run.

### Gap 2: Continuous self-learning has no demonstrated prospective utility

The store can replay, quarantine, restart, and roll back exact events. The V601 updater was safe but
did not beat read-only memory. FR11 still needs a prospective benefit on future rows, under frozen
commissioning choices, delayed correction, poison, retention, and an independent authority.

### Gap 3: The live ARC loop cannot yet assign causal credit to tools

The live `E3AgentPolicy` seam and first-party receipt schema exist. The current shared-context pool
can truncate the final channel, and its diagnostic clips the values needed to prove the cause.
Tool delivery cannot be interpreted until final-channel headroom is measured on the same live path.

## External Research Incorporated

- **Enoki** (`2609.00581`) now exposes a public 0.4B relation encoder and the EnokiQA dataset.
  Exp6886 admits a bounded cached shard and one shared anchored-relation schema. Exp6887 compares
  encoder, rule, and required GGUF proposal arms. None can approve its own records.
- **CoBRA** (`2609.00967`) treats tool activation as a counterfactual reward-margin problem.
  Exp6894 first collects replayable boundary rows. Exp6895 compares delivery and withholding from
  the same pre-action state and reports tool-favored, no-tool-favored, and ambiguous margins.
- **Parsing the Stream** (`2609.01466`) uses an append-only event ledger and typed compiled views.
  Exp6892 applies this pattern to complete ARC context, generation, supervisor, and action events.
  Raw-event replay, not a clipped view, is authority.
- **Cheap Verifiers, Large Blind Spots** (`2609.01345`) shows that in-loop metrics can improve while
  delivered error grows. Exp6891 and Exp6896 use independent exact checks and expose blind-spot
  rows. An in-loop-only success is `circular_positive`.
- **Retrieved but not ranked** (`2609.01556`) separates structural recall from surface-form rank.
  Exp6889 uses paraphrase-disjoint future rows and compares lexical with typed structural retrieval.
- **Leakage-free online adaptation** (`2609.01126`) freezes commissioning choices before the
  prospective stream. Exp6889 freezes thresholds; Exp6890 cannot tune on held future outcomes.

The dated source record, negative searches, and source URLs are in `research-references.md`, V603
Planner Refresh. MemoryWalker is retained as a future parametric-learning warning. No new KAN,
Ising, external scorer, constrained decoder, or hardware result closes a current blocker.

## Target Architecture

```text
             independent contract audit (Exp6885)
                          [advisory only]

  cached Enoki assets + exact fixture (Exp6886)
                       |
                       v
  rule + Enoki + three local GGUF proposal arms (Exp6887)
                       |
                       v
       sealed independent exact qualification (Exp6888)
                       |
                       v
       paraphrase-disjoint event stream (Exp6889)
                       |
        +--------------+----------------+----------------+
        |              |                |                |
    no memory    read-only lexical  read-only typed  bounded update
        |              |                |                |
        +--------- future exact outcomes and cost -------+
                       |
                       v
          sealed prospective audit (Exp6891)

  live E3AgentPolicy --> append-only trace contract (Exp6892)
                                      |
                                      v
                         context-pool headroom (Exp6893)
                                      |
                                      v
                      replayable tool boundaries (Exp6894)
                                      |
                                      v
                    deliver / withhold paired margin (Exp6895)

  independent blind-spot audit (Exp6896) --> cold capstone (Exp6897)
             [both run even when a branch is blocked]
```

The two science roots are independent. The relation fixture does not depend on the manifest audit.
The ARC trace repair does not depend on the relation branch. Exp6896 and Exp6897 preserve blocked,
null, partial, disqualified, circular-positive, and positive branch states without rerunning science.

## Model and Runtime Contract

Every task that performs new LLM inference must declare `MODEL_SPECS`, call
`cached_sota_pair()`, and use the native embedded GGUF tokenizer. It must use at least one of:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6887 uses all three. Exp6893-Exp6895 use the Qwen flagship on the canonical live path. A legacy
small model can run a CPU smoke test only. It cannot fill a headline cell. A missing model, native
tokenizer, CUDA offload receipt, or task-owned GPU lease produces a terminal blocked artifact.

## Phase A: Exact Relational Facts

### Exp6885: V603 executable-manifest and branch-isolation contract

Compare the design document, staged YAML, exclusion manifest, and exact prompt endings. Require 13
tasks in order, unique deliverables, valid structured gates, complete prior-failure blocks, and no
retired upstream. Prove that the manifest field is advisory and that Exp6886, Exp6892, Exp6896, and
Exp6897 have no global gate. This is infrastructure slot one.

**Deliverable:** `results/experiment_6885_v603_executable_manifest_branch_contract.json`  
**Advisory output:** `v603_manifest_contract_ready_score`

### Exp6886: Enoki asset admission and exact relation fixture

Admit pinned Enoki encoder and EnokiQA revisions into a bounded local cache. Define one versioned
record with source span, evidence span, normalized tuple, predicate vocabulary, and ASP atom. Build
calibration and sealed-held fixtures across supported bounded-ASP families. Prove atom and energy
parity against an independent solver. This task performs no GGUF inference.

**Deliverable:** `results/experiment_6886_enoki_exact_relation_fixture.json`  
**Gate output:** `relation_fixture_ready_score`

### Exp6887: Three-family relation proposal corpus

Run all three required GGUFs, the pinned Enoki encoder, and the deterministic rule arm on identical
source records. Preserve raw outputs, empty cells, parser failures, spans, tuples, model identity,
tokenizer evidence, GPU receipts, and latency. Keep sealed labels and solver sidecars unavailable.
Readiness means authentic complete acquisition, not extraction quality.

**Deliverable:** `results/experiment_6887_three_family_relation_proposal_corpus.json`  
**Gate output:** `relation_corpus_complete_score`

### Exp6888: Independent held relation qualification

Open held labels once in a fresh reducer. Compare rule, Enoki, and each GGUF proposal arm on source
span grounding, tuple accuracy, abstention, ASP compilation, and exact program semantics. Report
per-family and perturbation results. Because the exact executor is oracle authority, a passing
semantic class is `circular_positive`, never `positive`.

**Deliverable:** `results/experiment_6888_independent_relation_qualification.json`  
**Gate outputs:** `relation_qualification_ready_score`, `qualified_relation_event_count`

## Phase B: Prospective Structural Self-Learning

### Exp6889: Paraphrase-disjoint structural opportunity stream

Build a chronological stream from held-qualified relation rows and later exact outcomes. Freeze
commissioning thresholds before future rows. Measure whether lexical and typed structural retrieval
have candidate recall and downstream headroom under paraphrase, family transfer, conflicts, poison,
delay, restart, and rollback. No memory update occurs in this task.

**Deliverable:** `results/experiment_6889_paraphrase_disjoint_structural_stream.json`  
**Gate output:** `structural_stream_ready_score`

### Exp6890: Prospective structural-memory A/B

Run no-memory, read-only lexical, read-only typed, and bounded typed-update arms on identical frozen
orders. Update only after the exact outcome of an event. Report future exact utility, retrieval
position, admissions, rejections, latency, and bytes. This is the milestone's required continuous
self-learning experiment. A learning claim requires the update arm to beat the strongest read-only
arm; beating no memory alone is insufficient.

**Deliverable:** `results/experiment_6890_prospective_structural_memory_ab.json`  
**Gate output:** `structural_csl_run_complete_score`

### Exp6891: Sealed independent structural-learning audit

Cold-replay at least five frozen event orders. Recompute all comparisons from rows and exact
outcomes. Test poison, delayed correction, retention, paraphrase transfer, restart, persistence,
rollback, and update cost. The audit runs when Exp6890 completed even if its measured gain is zero.
A repeated no-gain-over-read-only result retires this structural-memory attempt.

**Deliverable:** `results/experiment_6891_sealed_structural_learning_audit.json`  
**Output:** `structural_csl_ready_score`

## Phase C: Context-Safe Live ARC Tool Margins

### Exp6892: Append-only ARC context and action trace contract

Replace clipped diagnostic text with raw typed events for prompt size, slot count, declared and
observed context, shared-pool capacity, requested and generated tokens, reasoning and final-channel
characters, supervisor calls, tool receipts, actions, states, and teardown. Compile existing views
from the event ledger and prove deterministic replay on the canonical scored-path harness. This task
does not run a live model or claim a solve.

**Deliverable:** `results/experiment_6892_arc_append_only_context_trace_contract.json`  
**Gate output:** `arc_trace_contract_ready_score`

### Exp6893: Live shared-context headroom sweep

Use adapter-disabled `E3AgentPolicy` on unseen generalization games. Hold prompt, model, seed,
generation, and action budgets fixed. Compare the current pool with a capacity computed from actual
prompt and generation receipts. Do not treat a larger `max_tokens` value as a repair. Readiness
requires measured final-channel headroom and reduced reasoning-only truncation, not a level solve.

**Deliverable:** `results/experiment_6893_live_arc_context_headroom_sweep.json`  
**Gate output:** `context_pool_headroom_ready_score`

### Exp6894: CoBRA-style replayable tool-boundary accrual

With context headroom qualified, collect first-party selfparse demands, available tools, refused and
accepted calls, same-state alternatives, actions, and exact next outcomes. Produce at least 12
replayable pre-action rows across three unseen games. A receipt alone is not an opportunity and is
not utility.

**Deliverable:** `results/experiment_6894_cobra_replayable_tool_boundaries.json`  
**Gate output:** `replayable_tool_boundary_row_count`

### Exp6895: Matched tool delivery-versus-withholding margin

Branch from each frozen pre-action state. Deliver the demanded tool in one arm and withhold it in
the other. Match model, prompt, context, seed, budgets, and pre-state hash. Credit only exact next
outcomes and actions-to-progress. Report tool-favored, no-tool-favored, and ambiguous pairs. No
game-level solve is a milestone claim.

**Deliverable:** `results/experiment_6895_matched_tool_delivery_margin.json`  
**Output:** `tool_loop_promotion_ready_score`

## Phase D: Independent Audit and Synthesis

### Exp6896: Cross-branch verifier-blind-spot audit

Read every available Phase A-C artifact and its raw rows. Compare in-loop scores with independent
exact outcomes, check row/headline consistency, replay event ledgers, and preserve missing or blocked
branches. This task is ungated and performs no new LLM inference. This is infrastructure slot two.

**Deliverable:** `results/experiment_6896_cross_branch_verifier_blind_spot_audit.json`  
**Output:** `v603_independent_audit_complete_score`

### Exp6897: Independent V603 capstone

Recompute branch dispositions from primary artifacts and conductor rows. Check model receipts,
field contracts, gates, retirement, exact authority, live ARC provenance, document/YAML parity, and
applicable end-to-end tests. Do not rerun science. The capstone is ungated, so every branch receives
a terminal record.

**Deliverable:** `results/experiment_6897_v603_independent_capstone.json`

## Dependency Graph

```text
Exp6885                                      [ungated advisory contract]

Exp6886 --> Exp6887 --> Exp6888 --> Exp6889 --> Exp6890 --> Exp6891

Exp6892 --> Exp6893 --> Exp6894 --> Exp6895

Exp6896 reads Exp6885..Exp6895 when present [ungated]
Exp6897 reads Exp6885..Exp6896 when present [ungated]
```

| Downstream task | Upstream field | Condition |
|---|---|---|
| Exp6887 | `exp6886.relation_fixture_ready_score` | `== 1` |
| Exp6888 | `exp6887.relation_corpus_complete_score` | `== 1` |
| Exp6889 | `exp6888.relation_qualification_ready_score` | `== 1` |
| Exp6889 | `exp6888.qualified_relation_event_count` | `>= 90` |
| Exp6890 | `exp6889.structural_stream_ready_score` | `== 1` |
| Exp6891 | `exp6890.structural_csl_run_complete_score` | `== 1` |
| Exp6893 | `exp6892.arc_trace_contract_ready_score` | `== 1` |
| Exp6894 | `exp6893.context_pool_headroom_ready_score` | `== 1` |
| Exp6895 | `exp6894.replayable_tool_boundary_row_count` | `>= 12` |

Every gate field appears under the upstream task's required artifact fields with the same spelling.
Every upstream task is in this roadmap. Exp6891 gates on completion, not positive learning utility,
so it can independently confirm a null. Exp6885, Exp6886, Exp6892, Exp6896, and Exp6897 are ungated.

## Failed-Scope and Retirement Boundaries

- Exp6874 blocked on the V602 document/YAML mismatch. Exp6885 checks a completed 13-task staging
  contract and does not gate science. It carries Exp6874 and retires on the same verdict.
- Exp6875 was a conductor gate block, not a fixture test. Exp6886 is an independent root, adds the
  public Enoki assets, and retains the qualified Exp6274 compiler. It carries Exp6875.
- Exp5786 failed its parser threshold, and Exp5923 retired schema-supported ConstraintIR decoding.
  Exp6887 uses source-anchored records, a closed predicate vocabulary, raw proposal rows, and no
  repair reprompt or constrained schema. It carries both failures.
- Exp5909 found no exact gain from structured prompt synthesis. Exp6888 tests held grounded
  extraction and exact compilation, not prompt repair. It carries Exp5909 and Exp5923.
- Exp5773 ended on an unready prospective constraint stream. Exp6890 uses held-qualified relation
  events, frozen commissioning, and typed structural retrieval. It carries Exp5773 and Exp6873.
- Exp6873 found no update gain over read-only. Exp6891 tests a different learned object and retires
  this attempt if the same no-gain verdict recurs.
- Exp6844 had zero outcome-credit headroom. Exp6893 changes the prerequisite with raw context-pool
  events and an actual capacity sweep before it interprets outcomes.
- Exp6845 had zero first-party tool-gap obligations. Exp6894 and Exp6895 require a headroom-qualified
  live run, Exp6859's receipt seam, and replayable same-state alternatives. They carry Exp6845.

No task reuses a retired experiment ID. No structured gate or dependency names a retired task.
Generated-text external energy scoring, scalar semantic likelihood, schema-supported ConstraintIR
reprompting, and the V601 reliability updater remain closed.

## Hardware Requirements

| Resource | Use | Claim boundary |
|---|---|---|
| Two RTX 3090 GPUs | Exp6887 local three-family acquisition; Exp6893-Exp6895 live ARC runs | Record process, GPU UUID, placement, offload layers, VRAM, lease, and teardown for each model cell. |
| Local CPU and RAM | Exact ASP checks, Enoki cache reduction, structural stream, replay, audits | Use explicit deterministic or encoder-only substrate names. Do not inherit live-GGUF duration claims. |
| Local model cache | All three required GGUFs plus pinned Enoki encoder and bounded EnokiQA shard | Record revisions and content hashes. No silent network or tiny-model substitution. |
| KV260, GateMate, PolarFire | Outside the blocking graph | Existing terminal receipts stand. Schedule no repeated board task without new physical evidence. |
| Extropic Z1 / TSU | Not locally available | Make no execution, speed, power, throughput, or access claim. |

If a required cache asset or exclusive GPU lease is unavailable, the experiment writes a terminal
blocked artifact with `gate_check_summary`. It does not replace a headline model with a legacy CPU
smoke model.

## Experimental Validity and Claim Rules

- Add the task's `REQ-*` and scenarios before implementation code.
- Compute preconditions before expensive work. Preserve a terminal blocked artifact on failure.
- Every artifact declares `inference_substrate`, `verdict_class`, `honest_verdict`, and one
  `field_principles` annotation for every required field and gate field.
- Every comparison emits one `rows` entry per model, fixture, event, arm, order, game, seed, and
  condition. Headline values must recompute from those rows.
- A blocked verdict records `gate_check_summary` with the failed check, expected value, and observed
  value. Do not invent a second diagnostics field.
- `verifier_is_oracle=true` forbids `verdict_class=positive`; use `circular_positive` when all
  declared checks pass under the same exact authority.
- Continuous learning cannot read future labels or tune after the prospective stream starts. It
  must beat the strongest read-only arm and pass safety checks before promotion.
- The ARC live path is `E3AgentPolicy` through `arc_scored_path_lever_harness`, with adapters off.
  ARC work does not read game source, use offline BFS, build a per-game adapter, or claim a solve.
- Any incidental level advance records `solve_provenance=live_agent_self_discovery` and remains
  uncredited as a milestone solve. Exp6892-Exp6895 make no level-solve claim.
- Do not modify `scripts/research_conductor.py` and do not push.

## Milestone Exit Criteria

V603 succeeds as a research milestone when all 13 tasks reach terminal conductor states and the
ungated audit and capstone preserve each branch's evidence. Scientific results can be positive,
circular-positive, null, blocked, disqualified, or partial.

A relation claim requires authentic proposal-arm receipts, sealed held reduction, span grounding,
and exact ASP parity. A self-learning claim requires prospective update benefit over both read-only
controls on the replication rule with zero safety failures. A live tool claim requires measured
context headroom, replayable same-state opportunities, and positive exact paired margins. No pooled
headline can override its per-unit rows.

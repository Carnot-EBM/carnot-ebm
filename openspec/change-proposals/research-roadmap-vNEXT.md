# Research Roadmap vNEXT: V627

**Milestone:** `2026.09.627`

**Title:** Source-grounded verification, verifier-balanced self-learning, and non-degenerate live-path evidence

**Status:** Pre-staged next milestone

**Task contract:** Exactly 12 tasks, `exp7136` through `exp7147`, in the order below.

## What V626 proved

Milestone `2026.09.626` completed its exact 12-task contract. Its positive
sampling result and its negative verifier, learning, and ARC evidence define a
narrow next step.

| Evidence | Result | Meaning for V627 |
|---|---|---|
| Exp7124-Exp7125 | The Markdown/YAML contract matched. All three mandated Q4 GGUFs remained cached. | Preserve the literal contract check and repeat source/cache checks at execution time. |
| Exp7127-Exp7128 | Both ARC arms completed only three actions and zero levels. Every real generation reached the 384-token cap. The audit also found target-adapter code in the withheld process. | Reuse the same driver, raise action and generation limits, fix process isolation, and disqualify another degenerate control. |
| Exp7129-Exp7130 | A three-family exact constraint bank ran, but its free-text substrate alias was invalid. The clean routing study accepted no exact errors, but produced no useful retries or promoted rejected actions. | Do not repeat confidence routing on the same bank. Move to source-grounded relational verification with an independently scored intervention. |
| Exp7131-Exp7132 | The model-facing continuous-learning task produced no artifact after two timeouts and an incomplete module. Its portability audit was cascade-blocked. | Materialize the event stream first. Test one bounded external-memory update method, then audit its stored result without new inference. |
| Exp7133-Exp7134 | The corrected multiscale proposal passed exact parity and improved matched-budget host ESS. | Port the proved method to Rust. Require exact parity before any throughput claim. |
| Exp7135 | Only the sampling branch was positive. Contract and source work were complete; ARC and routing were disqualified or null; learning and portability were missing. | V627 must close evidence quality, not add broad new branches. |

V623 remains the positive self-learning control. Exp7106 found delayed
procedural-memory value on a sealed synthetic stream. It did not test a current
GGUF model inside the online loop.

## Three largest gaps to the PRD vision

1. **Useful source-grounded verification is not established.** Exact outcome
   checks work, but the old facts scorer was domain-bound and V626 uncertainty
   routing produced no useful intervention. Carnot needs a verifier that
   changes decisions on source-backed claims and is scored by hidden external
   labels.
2. **Continuous self-learning is not model-facing.** Transactional memory and
   a positive synthetic learning result exist. The current SOTA GGUF path has
   no completed chronological learn-then-act result, no stale-memory control,
   and no cold retention audit.
3. **ARC generalization evidence is still degenerate.** The live E3 path has
   public per-game solves, but its adapter-withheld experiment had a zero-level
   positive control, hard truncation, and failed process isolation. This does
   not measure withheld-game value.

The Rust sampler port and GateMate continuity task support the PRD's production
and hardware paths. They do not replace these three scientific gaps.

## Research basis

The dated `V627 Planner Refresh` in `research-references.md` records the source
sweep before this design was written.

- FlowBalance (`arXiv:2609.03241`) motivates a signed external-memory update:
  retain guidance after positive verifier advantage, reverse it after negative
  advantage, and make no update when a group has no preference. V627 keeps all
  GGUF weights frozen and does not claim to reproduce FlowBalance training.
- Low-level symbolic grounding (`arXiv:2609.05025`) motivates a source-to-SQL
  hallucination detector. V627 treats model-written relations and SQL as
  untrusted proposals. A sandboxed SQL executor and hidden RAGTruth labels
  remain the authorities.
- MentorPulse (`arXiv:2608.20927`) motivates bounded refresh and an explicit
  stale-memory control. Its latent cross-attention architecture is not
  available through the current GGUF runtime.
- The ICML 2026 long-horizon study motivates an ARC action budget large enough
  for a positive control to act. A larger ceiling is not an efficiency result.
- Recent Geo-LoRA and KAN-adapter work changes trainable parameters. It does
  not reopen Carnot's retired PWA-KAN or within-chain adaptation lineages.
- Extropic Z1T and Logical Intelligence Kona remain unattached vendor or
  product comparators. Neither is executable evidence for V627.

## Target architecture

```text
 source passage + candidate response
                  |
                  v
        mandated local GGUFs
   Qwen3.6 MoE / Gemma-4 dense / Gemma-4 MoE
          | direct | self-check | relations + SQL
          v
 untrusted structured proposal ---> sandboxed SQL execution
          |                                  |
          +---------- hidden labels <--------+
                            |
                      exact later outcome
                            v
 immutable event stream -> verifier-balanced external memory
        |                     | positive: retain
        |                     | negative: reverse
        |                     | tie: no update
        |                     v
        +-------------- future held-out model actions
                               |
                    cold retention / stale audit

 public ARC registry -> reused adapter-withheld E3 driver -> executed actions
                              |                         |
                    larger bounded budgets       transition receipts
                              |                         |
                              +---- isolation audit ---+

 proved Python multiscale sampler -> exact Rust parity -> host throughput
 new GateMate physical receipt? -> one detect/smoke action or terminal block
```

The architecture preserves the exact-authority boundary. Models propose facts,
relations, queries, actions, and memory candidates. Hidden labels, SQL
execution, exact environment transitions, and exact finite laws score them.
Only later verified outcomes may change external memory.

## Exact task contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7136-v627-contract-preflight` | V627 Markdown and YAML task-contract preflight | `results/experiment_7136_v627_contract_preflight.json` | none |
| 2 | `exp7137-v627-source-and-cache-delta` | V627 execution-time source and SOTA cache delta | `results/experiment_7137_v627_source_delta.json` | none |
| 3 | `exp7138-source-grounded-relational-fixture` | Source-grounded relational hallucination fixture | `results/experiment_7138_v627_relational_fixture.json` | none |
| 4 | `exp7139-three-family-symbolic-grounding-ab` | Three-family symbolic grounding comparison | `results/experiment_7139_v627_symbolic_grounding_ab.json` | `exp7138-source-grounded-relational-fixture.source_grounding_fixture_ready_score == 1` |
| 5 | `exp7140-symbolic-intervention-causal-audit` | Independent symbolic intervention and causal audit | `results/experiment_7140_v627_symbolic_intervention_audit.json` | `exp7139-three-family-symbolic-grounding-ab.symbolic_grounding_complete_score == 1` |
| 6 | `exp7141-v626-chronological-csl-stream` | Immutable V626 chronological self-learning stream | `results/experiment_7141_v627_csl_event_stream.json` | none |
| 7 | `exp7142-flowbalance-external-memory-csl` | Verifier-balanced external-memory continuous self-learning | `results/experiment_7142_v627_flowbalance_memory_csl.json` | `exp7141-v626-chronological-csl-stream.csl_event_stream_ready_score == 1` |
| 8 | `exp7143-flowbalance-memory-cold-audit` | Cold retention and negative-transfer audit | `results/experiment_7143_v627_flowbalance_memory_cold_audit.json` | `exp7142-flowbalance-external-memory-csl.flowbalance_memory_csl_complete_score == 1` |
| 9 | `exp7144-rebudgeted-adapter-withheld-arc-loo` | Rebudgeted adapter-withheld ARC LOO cell | `results/experiment_7144_v627_rebudgeted_arc_loo.json` | none |
| 10 | `exp7145-rust-multiscale-sampler-parity` | Rust multiscale sampler parity and throughput | `results/experiment_7145_v627_rust_multiscale_sampler.json` | none |
| 11 | `exp7146-gatemate-changed-state-continuity` | GateMate changed-state continuity with one-action stop | `results/experiment_7146_v627_gatemate_changed_state.json` | none |
| 12 | `exp7147-v627-capstone` | V627 independent evidence matrix and branch disposition | `results/experiment_7147_v627_capstone.json` | none |

No other task belongs to V627. The YAML must contain these 12 full IDs, titles,
deliverables, and gates in this exact order.

## Phase 0: Contract and current source state

### Exp7136: V627 Markdown and YAML task-contract preflight

Parse the Markdown and YAML independently. Compare the 12 rows, their order,
titles, deliverables, structured gates, producer fields, model declarations,
failure-history records, focused tests, and prompt tails. The task also runs
roadmap schema, exclusion-manifest, gate, harness-fit, and contract checks. It
is advisory so a planner defect is visible without cascade-blocking science.

This is a recurring contract scope. It declares Exp7109 and Exp7121 as prior
failures. The new attempt uses the exact activated YAML rather than a staged
subset, and it retains mechanical retirement if the same verdict returns.

### Exp7137: V627 execution-time source and SOTA cache delta

Repeat the requested primary and secondary source checks when the task runs.
Append one idempotent dated execution delta. Verify all three mandated model
repositories, exact cached GGUF paths, revisions, sizes, and hashes without a
download. Separate first-party research, vendor claims, product claims, and
community discovery. This task records a source delta; it does not claim an
experimental result from a paper.

## Phase 1: Source-grounded verification

### Exp7138: Source-grounded relational hallucination fixture

Freeze at least 72 balanced cached RAGTruth rows by source group before outcome
inspection. Store source passages and candidate responses in the model view.
Keep span and response labels in a sealed scorer view. Define a strict relation
schema, a read-only SQLite query contract, time and row limits, and forbidden
SQL operations. An independent loader must reproduce every split, label, and
hash. This task makes no verifier-value claim.

The conservative scope matcher links this fixture to Exp6984. V627 declares
that prior result and distinguishes external source labels plus a SQL sandbox
from the earlier exact mapping fixture.

### Exp7139: Three-family symbolic grounding comparison

Run all three mandatory GGUF families with correct chat templates. Compare a
one-pass direct detector, a matched two-pass self-verifier, and a matched
two-pass SQL-grounded detector. In the SQL arm, the model extracts relations
and writes a bounded query. A sandbox executes the query. Hidden RAGTruth
labels score all arms. Report every row, prompt, output, parse, query result,
invalid-query reason, latency, tokens, and cost by model and source family.

The primary claim is per-family useful detection at matched opportunity, not a
pooled win. The model-written database and query are not an oracle. This is a
new symbolic grounding mechanism, not a rerun of the retired generated-text
facts scorer.

### Exp7140: Independent symbolic intervention and causal audit

Use no new model calls. Recompute metrics from raw rows and sealed labels.
Verify source blindness, SQL sandbox limits, template and budget parity, and
model identity. Remove the SQL execution result from each SQL-grounded
decision and replay the final decision rule. Count useful, harmful, and null
interventions per model and source family. A completed fixture is not a
positive verifier result unless these rows support it.

## Phase 2: Continuous self-learning

### Exp7141: Immutable V626 chronological self-learning stream

Read Exp7130 raw manifests and receipts directly. Do not promote Exp7129's
flagged headline. Build at least 96 chronological events stratified by model,
constraint family, and hardness. Seal past, adaptation, future, and protected
retention splits. The event view withholds exact outcomes until after its
action receipt. An independent loader must reproduce ordering, hashes, and
labels. This task makes no learning claim.

The task declares the blocked Exp6265 and Exp6277 chronological-learning
lineages. It removes their unavailable gates and produces only a sealed input
stream for a later task.

### Exp7142: Verifier-balanced external-memory continuous self-learning

This task satisfies the milestone's continuous self-learning requirement.
Use a bounded Qwen3.6 panel and frozen weights. Compare no memory, equal-budget
raw trace, delayed procedural memory, and verifier-balanced strategy memory.
After a later exact outcome, retain guidance for positive advantage, reverse
it for negative advantage, and make no update for tied groups. All writes use
transactional staging, signatures, rollback, and fixed schemas. Memory from an
episode may affect future events only.

Report per-event later exact success, learning curves, abstention, conflicts,
stale-memory harm, protected retention, recovery, latency, tokens, and cost.
Completion and uplift are separate fields. The task declares the null Exp6978
lineage and the missing Exp7131 artifact as prior failures. The new method has
a prepared event stream and a narrower one-model loop.

### Exp7143: Cold retention and negative-transfer audit

Use no new model calls. Recompute every admitted memory transaction and result
from Exp7142 raw receipts. Restart from stored files, delete one source record,
inject one stale record, inject one conflict, corrupt one signature, and test
rollback. Report protected retention and negative transfer by source family,
hardness, and time segment. External upstream failure is `blocked`, not
`partial`.

The task declares the blocked Exp6963 cold-audit lineage. Its same-milestone
producer field replaces the unavailable queue-memory gate.

## Phase 3: Live-path, sampler, hardware, and synthesis

### Exp7144: Rebudgeted adapter-withheld ARC LOO cell

Reuse `python/carnot/experiment_7127_v626_adapter_withheld_arc_loo.py`; do not
create a fifth LOO harness. Add flags while preserving its V626 defaults. Run
the new cell with 25 actions and 1,024 generation tokens. Fix worker import
order so withheld target-adapter code is not loaded. Select registry
eligibility rank one before new outcomes. Use cached Qwen3.6, fresh processes,
the real `E3AgentPolicy` and `make_carnot_agent` path, and executed environment
actions.

Disqualify the cell if process isolation fails, the visible control reaches no
level, all real outputs reach the new token cap, or the arms have no measured
input or policy difference. A clean zero delta with a working control is a
terminal null. The result is `development_proxy`, does not update the solve
registry, and cannot claim a hidden-game or game-level solve. Exp7127 and
Exp7128 are declared prior failures, with the changed limits and isolation as
the new condition.

### Exp7145: Rust multiscale sampler parity and throughput

Port the proved Exp7133 proposal and Exp7134 correction into
`crates/carnot-samplers`. Use identical fixtures, seeds, proposals, accept
decisions, chains, energies, and finite laws across Python and Rust. Exact
distribution and replay parity must pass before speed is reported. Benchmark
matched host work with confidence intervals. Report any sub-10x result
honestly. Do not claim FPGA, TSU, WCRG, or asymptotic acceleration.

The task declares Exp5714 as the earlier one-axis Rust parity lineage. V627
changes the algorithm to the newly proved multiscale proposal and adds matched
throughput only after exact shared-randomness parity.

### Exp7146: GateMate changed-state continuity with one-action stop

First require a new operator-authored physical-state receipt newer than
Exp6559. If it is absent, write one terminal blocked artifact and run no JTAG
command. If it exists, run one `openFPGALoader -c dirtyJtag --detect` action.
Only after the expected identity appears may the task run the existing n=16
smoke. Stop on the first failed action. Do not redesign a bitstream and do not
claim speed. KV260 and PolarFire are terminal and are not repeated.

This required hardware-continuity task declares Exp6525 and Exp6559 as prior
blocks. The physical receipt is the only allowed changed condition.

### Exp7147: V627 independent evidence matrix and branch disposition

Read the other 11 artifacts by exact path. Recompute gate values, row counts,
headline metrics, hashes, and verdict classes. Separate completion from
scientific promotion. Give each branch one disposition: promote, continue,
retire, repair, or blocked pending external state. Mark absent or externally
blocked inputs as `blocked`, not `partial`. Do not rerun inference or hardware.

The capstone declares Exp6922's partial-result lineage. It uses `blocked` for
unchanged absent or external inputs and does not spend retries on them.

## Dependency graph

```text
exp7136  contract preflight (advisory)       exp7137 source/cache delta

exp7138 relational fixture
   └──[source_grounding_fixture_ready_score == 1]──> exp7139 symbolic A/B
          └──[symbolic_grounding_complete_score == 1]──> exp7140 causal audit

exp7141 chronological CSL stream
   └──[csl_event_stream_ready_score == 1]──> exp7142 external-memory CSL
          └──[flowbalance_memory_csl_complete_score == 1]──> exp7143 cold audit

exp7144 rebudgeted ARC cell       exp7145 Rust sampler       exp7146 GateMate
               \                       |                       /
                +----------------------v----------------------+
                                  exp7147 capstone
```

The capstone has no structured gate. It must record missing or blocked inputs
without spending retries on unchanged external conditions.

## Hardware and model requirements

| Task | Compute or hardware | Required boundary |
|---|---|---|
| Exp7136, Exp7137, Exp7138, Exp7140, Exp7141, Exp7143, Exp7147 | CPU, repository, and stored artifacts | No model download and no new inference unless the task states it. |
| Exp7139 | Local GPU and the three cached Q4 GGUFs | Use Qwen3.6-35B-A3B, Gemma-4-31B-it, and Gemma-4-26B-A4B with correct templates. Print per-model progress. |
| Exp7142 | Local GPU and cached Qwen3.6-35B-A3B Q4 | Frozen weights; bounded future panel; transactional external memory only. |
| Exp7144 | Local GPU, cached Qwen3.6-35B-A3B Q4, and live E3 runtime | Fresh worker processes, 25-action cap, 1,024-token cap, no registry mutation. |
| Exp7145 | Host CPU and Rust toolchain | Exact parity precedes throughput. No accelerator claim. |
| Exp7146 | GateMate A1 only after a new physical receipt | One detect action, then the existing n=16 smoke only if identity matches. Stop on first failure. |

No attached Extropic TSU is available. KV260 and PolarFire already have
terminal continuity evidence. V627 performs no FPGA bitstream redesign and no
unattached-hardware benchmark.

## Milestone success conditions

- The Markdown and YAML contain the same 12 tasks, in the same order, with the
  same deliverables and structured gates.
- At least 72 source-grounded rows are independently reproducible, and the
  three-family symbolic comparison exposes every per-unit result.
- The continuous self-learning branch completes a chronological frozen-model
  loop or produces an honest terminal result with no missing artifact.
- The ARC cell is informative only if its control, generation budget, arm
  distinction, and process isolation pass.
- Rust matches the sealed Python sampler before a throughput result is named.
- GateMate performs no command without new physical state.
- Every artifact declares `verdict_class`; every comparison emits per-unit
  rows; every blocked result names its failed check in `gate_check_summary`.

## Explicit deferrals

- No new KAN, PWA-KAN, LoRA, external-text scorer, grammar, FSNet, or
  within-chain activation experiment.
- No new ARC per-game adapter, offline ground-truth BFS, public registry
  re-solve, or hidden-game solve claim.
- No learned verifier may replace SQL execution, hidden labels, exact
  constraints, environment transitions, or finite-law checks.
- No Gemma-to-Qwen portability matrix before the one-model CSL task and cold
  audit complete.
- No FPGA redesign, Extropic runtime claim, Kona baseline claim, or scaling
  claim from a host-only benchmark.

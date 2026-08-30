# Carnot Research Roadmap vNEXT: Exact Repair, Active Memory, and ARC Progress Evidence

**Milestone:** `2026.08.590`  
**Created:** 2026-08-30  
**Status:** Proposed  
**Supersedes:** milestone `2026.08.589`  
**Research basis:** `research-program.md`, `_bmad/prd.md`, `_bmad/architecture.md`,
`ops/status.md`, `ops/changelog.md`, `research-complete.yaml`, `research-roadmap.yaml`, prior
roadmap proposals, `ops/conductor-log.md`, `research-hardware-wishlist.md`, and the
`V590 Planner Refresh` in `research-references.md`.

## What milestone 2026.08.589 proved

Milestone `.589` reached a terminal state for all thirteen planned task positions. Its capstone
correctly reported a partial result. The useful evidence is narrow and reproducible.

| Branch | What `.589` proved | Remaining boundary |
|---|---|---|
| Proof transport | Exp6755 losslessly replayed 216/216 local GGUF outputs. Eleven rows were exact-valid. The transport bug is closed. | Only 21 rows fit the frozen environment-grammar target classes. The gate required 24, so the grammar, live A/B, audit, diagnostic energy, and repair did not run. |
| Continuous self-learning | Exp6761 produced six chronological orders with matched trace and procedural representations. Accepts, rejects, restart, rollback, and poison fixtures all passed. | Exp6762 did not run the prospective A/B. Its owned checks found no one-model VRAM admission and no task-owned lease. Exp6763 therefore gate-blocked. |
| ARC live path | Exp6764 loaded Qwen3.8 and Qwen3.6 one at a time on a leased RTX 3090. It recorded first-token, selfparse dispatch, teardown, and VRAM recovery receipts. | Exp6765 rejected the upstream artifact because it required `model_specs` while Exp6764 emitted `models_used`. No ARC quality comparison ran. Separately, the shipped trajectory supervisor, tool-gap capture, and selfparse tool loop still lack the evidence needed for adoption. |
| Hardware portability | Exp6766 independently reproduced the context-matching trajectory reduction. | Trajectory refinement still optimizes the same exact objective used for evaluation. Its strongest result is circular and simulator-only. There is no physical TSU claim. |
| Milestone synthesis | Exp6767 preserved blocked, partial, circular, and positive branch states without pooling them. | FR12 exact repair, FR11 prospective learning, and live ARC action efficiency remain open. |

The next step is not another broad architecture change. It is to repair the three broken evidence
contracts and run the comparisons that `.589` could not reach.

## The three largest gaps to the PRD vision

| Rank | PRD gap | Current evidence | `.590` response |
|---|---|---|---|
| 1 | **FR12 can certify a proof but cannot reliably produce or repair exact certificates.** | Lossless transport gives 11/216 exact-valid rows. The grammar chain stopped at 21 targetable errors, three below an arbitrary panel floor. No exact repair result exists. | Expand the panel with exact-preserving counterfactuals. Build an instance-bound runtime grammar. Compare direct, static, and draft-conditioned decoding on all three mandated GGUF families. Cold-audit the result, then test claim-localized prefix backtracking. |
| 2 | **FR11 has transactional storage but no prospective continuous self-learning result.** | The stream and transaction mechanics are ready. The actual procedural-versus-trace A/B never acquired a model lease. | Prove a task-owned one-model-at-a-time lease and artifact contract. Run the six-order prospective comparison with actual retrieval and use. Audit retention, hard-case harm, poison, restart, and rollback in a fresh process. |
| 3 | **The live ARC agent has default-off tools and supervision without actions-to-progress evidence.** | Selfparse transport passed 20/20. The supervisor ledger is below its evidence floor. No artifact carries live `tool_gap_events`. No paired result tests whether selfparse reduces actions to level-up. | Accrue shadow-supervisor evidence at window 120. Prove tool-gap events reach an artifact and the refinement tool. Fix arm isolation. Run control-unset versus selfparse on actions to progress, then cold-audit the adoption decision. |

## Research findings used by this milestone

- **Decode-Time Grammars** (`arXiv:2607.18357`) provides the instance-bound symbol and prefix-state
  mechanism for the proof grammar.
- **Draft-Conditioned Constrained Decoding** (`arXiv:2603.03305`) keeps free semantic planning
  separate from structural enforcement. The experiment measures exact validity, not syntax alone.
- **Project Aletheia** (`arXiv:2601.14290`) motivates explicit conflict detection and bounded
  backtracking. `.590` uses runtime grammar conflicts, not an answer-conditioned verifier, to pick
  a repair region.
- **Claim-Level Reliability Assessment** (`arXiv:2608.11994`) motivates the smallest responsible
  proof-region receipt. It does not replace exact authority.
- **When Continual Learning Moves to Memory** (`arXiv:2604.27003`) and **Harness Continual
  Learning** (`arXiv:2608.19013`) motivate prospective order, equal capacity, actual retrieval,
  candidate acceptance, and historical-loss checks.
- Extropic's 2026 Z1 update remains a future hardware signal. Public device access is planned for
  2027. This milestone makes no TSU latency, power, or availability claim.
- Kona 1.0 remains architecture context. It provides no public weights or local runner and is not
  an executable baseline.

The literature refresh is recorded in `research-references.md` before this roadmap. `.590` is a
continuation milestone, so it does not spend another experiment slot on a duplicate SOTA sweep.

## vNEXT architecture

```text
                       EXACT AUTHORITY BOUNDARY
              proposal mechanisms never receive answer labels

 Frozen proof rows ──► targetable-panel expansion ──► runtime proof grammar
        │                                                   │
        │                                     direct / static / DCCD
        │                                                   ▼
        └──────────────────────────────────────────► proof candidates
                                                            │
                                         cold parser + exact checker
                                                            │
                                    conflict region ──► prefix backtrack
                                                            │
                                                    final exact check

 Frozen chronological stream ──► task-owned GGUF lease and schema receipt
        │                                                   │
        ▼                                                   ▼
 read-only episode ──► no memory / trace memory / procedural memory
        │                                                   │
        └── exact admission between episodes ◄──────────────┘
                              │
                 cold retention / poison / rollback audit

 Live E3AgentPolicy ──► shadow supervisor at window 120 ──► refinement ledger
        │
        ├── selfparse run ──► tool_gap_events ──► tool-gap refinement
        │
        └── isolated control-unset vs selfparse ──► actions to progress
                                                       │
                                              cold adoption audit
```

The exact checker certifies final proof rows and memory admissions. It may not select a generated
answer or enter a proposal feature vector. The ARC branch uses the production `E3AgentPolicy` /
`make_carnot_agent` route. It does not inspect game source, run an offline ground-truth BFS, add a
per-game adapter, or claim a new public-game solve. Public level-ups are development-proxy rows.

## Phase 1: Recover exact proof generation and repair

### Exp 6768: Exact-invalid targetable proof panel expansion

Start from the 21 targetable Exp6755 rows. Generate exact-preserving counterfactual error variants
for undefined variables, invalid clauses, non-binary values, duplicates, missing evidence, and
premature terminals. Preserve the source problem, source output hash, mutation operator, and exact
failure receipt. The task must produce at least 36 parseable exact-invalid rows across every held
family without reading an answer during mutation.

**Deliverable:** `results/experiment_6768_targetable_proof_panel_expansion.json`

### Exp 6769: Environment-indexed proof grammar fixture v2

Build the runtime grammar over the expanded panel. Bind variables, clause IDs, domains, uniqueness,
and remaining required slots to the current problem. Prove that valid SAT and UNSAT certificates
remain reachable and that ghost references are unreachable. Record actual runtime mask calls. A
post-hoc filter does not pass.

**Deliverable:** `results/experiment_6769_environment_indexed_proof_grammar_v2.json`

### Exp 6770: Three-model DCCD environment-grammar A/B v2

Run repaired direct output, static grammar, and draft-conditioned environment grammar on the same
frozen panel with Qwen3.6-35B-A3B, Gemma-4-31B, and Gemma-4-26B-A4B. Match total generation tokens,
context, seeds, and exact-check budgets. The headline is paired exact-valid yield. Parseability is a
secondary metric.

**Deliverable:** `results/experiment_6770_dccd_environment_grammar_ab_v2.json`

### Exp 6771: Independent proof transport and localization audit

Use a cold parser, checker, and reducer that do not import the producer's parsing, checking, or
aggregation functions. Recompute every arm result. Audit runtime invocation, budget equality,
answer leakage, support contraction, proof-preserving relabeling, and the smallest responsible
grammar-conflict region.

**Deliverable:** `results/experiment_6771_proof_transport_localization_audit.json`

### Exp 6772: Claim-localized prefix-backtracking repair A/B

Compare no repair, matched full regeneration, and bounded prefix backtracking from the audited
grammar-conflict region. Use Qwen3.6-35B-A3B and Gemma-4-31B. The repair selector may use only
pre-oracle grammar and certificate structure. Exact authority checks the final candidate. Positive
credit requires a paired gain over full regeneration without more harmful flips or support loss.

**Deliverable:** `results/experiment_6772_claim_localized_prefix_backtracking_ab.json`

## Phase 2: Run prospective continuous self-learning

### Exp 6773: Task-owned SOTA memory lease and artifact contract

Validate the ready Exp6761 stream, then load Qwen3.6-35B-A3B and Gemma-4-31B one at a time under the
existing receipt-scoped GPU lease. Record exact `model_specs`, first-token inference, peak VRAM,
teardown, VRAM recovery, and stream hashes. This task repairs the two failed Exp6762 checks and the
`model_specs` contract mismatch. It makes no learning claim.

**Deliverable:** `results/experiment_6773_csl_owned_lease_contract.json`

### Exp 6774: Procedural versus trace memory prospective A/B v2

Run no-memory, detailed-trace, and procedural-memory arms across all six frozen event orders. Keep
each episode read-only. Apply exact-approved transactions only between episodes. Match storage,
top-k, context, decode, and update budgets. Measure prequential yield, hard-case yield, retention,
forgetting, support, commits, rejects, actual retrieval, and action influence.

**Deliverable:** `results/experiment_6774_procedural_vs_trace_csl_ab_v2.json`

### Exp 6775: Independent continuous-learning durability audit v2

Recompute Exp6774 from raw rows and state receipts in a fresh process. Check chronology, capacity,
actual memory use, order-level intervals, hard-case harm, historical loss, poison, restart, and
byte-exact rollback. Stored but unused lessons do not support a learning claim.

**Deliverable:** `results/experiment_6775_csl_durability_audit_v2.json`

## Phase 3: Produce ARC supervisor and tool-loop evidence

### Exp 6776: Window-120 shadow-supervisor evidence accrual

Run the production live path with the trajectory supervisor in shadow mode and a window of 120.
Checkpoint each cell and install the long-run death receipt. Accrue the missing firings toward ten
per arm, then run `scripts/arc_supervisor_refine.py`. Shadow mode must not change scored actions.

**Deliverable:** `results/experiment_6776_arc_shadow_supervisor_accrual.json`

### Exp 6777: Live selfparse tool-gap transport receipt

Run one bounded production selfparse session. Prove that `tool_gap_events`, including an empty list,
survives from the live induction loop to the result artifact and that `scripts/arc_tool_gap_refine.py`
ingests the artifact. Observing a real gap is useful but is not required for a transport pass.

**Deliverable:** `results/experiment_6777_arc_tool_gap_transport.json`

### Exp 6778: Selfparse actions-to-progress A/B

Extend arm isolation so `CARNOT_ARC_INDUCE_TOOL_LOOP` is saved, set, and restored. Run the production
live path with control truly unset and treatment set to `selfparse`. Pair games, seeds, action
budgets, and model hashes. Measure actions to level-up and no-progress censoring. Record
memorization as an observation, not a penalty. Do not award a new solve.

**Deliverable:** `results/experiment_6778_arc_selfparse_actions_to_progress_ab.json`

### Exp 6779: Independent ARC adoption and refinement audit

Cold-recompute the actions-to-progress comparison, verify environment isolation, replay level-up
receipts, confirm tool-gap ingestion, and inspect the updated supervisor recommendation. Recommend
promote, retain-default-off, or retire. The task must not enable a flag or change the submission
kernel.

**Deliverable:** `results/experiment_6779_arc_tool_supervisor_adoption_audit.json`

## Phase 4: Milestone disposition

### Exp 6780: V590 branch disposition and PRD gap update

Run after every other task is terminal. It is intentionally ungated. Recompute comparative claims
from rows, preserve missing and blocked artifacts, and classify the FR12, FR11, and ARC gaps as
narrowed, unchanged, widened, or blocked. Reconcile the evidence and operations documents without
activating a successor roadmap.

**Deliverable:** `results/experiment_6780_v590_branch_disposition.json`

## Dependency graph

```text
Proof:
  6768 ──► 6769 ──► 6770 ──► 6771 ──► 6772

Continuous self-learning:
  6773 ──► 6774 ──► 6775

ARC live evidence:
  6776 ──► 6777 ──► 6778 ──► 6779

Milestone close:
  6768..6779 terminal ──► 6780 (ungated synthesis)
```

No branch gates another branch. A proof failure cannot suppress memory or ARC. A GPU admission
failure in memory cannot suppress proof work. The capstone records every terminal state.

## Hardware and runtime requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| Two RTX 3090 GPUs | 6770, 6772, 6773, 6774, 6776-6778 | Use task-owned receipt-scoped leases. Load one GGUF per GPU session unless an ARC task explicitly isolates one arm per card. Do not signal unrelated processes. Record device UUID, model hash, CUDA offload, peak VRAM, teardown, and recovery. |
| Local GGUF cache | 6770, 6772, 6773, 6774 | Required headline models are `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and, for Exp6770, `unsloth/gemma-4-26B-A4B-it-GGUF`. Embedded tokenizers and exact files must resolve before a run. |
| ARC generator | 6776-6778 | Keep `unsloth/Qwen3.8-27B-GGUF` as the immutable scored ARC generator. Include Qwen3.6-35B-A3B as the mandated transport canary and keep its rows out of Qwen3.8 quality reducers. |
| CPU, RAM, disk | All | Use exact solvers, cold reducers, and atomic JSON writes locally. Preserve enough RAM and disk for one model, sharded ARC checkpoints, raw rows, and state journals. |
| KV260 | None | Its terminal latency transcript and synthesis flag already satisfy the continuity terminal state. No new bitstream redesign is allowed. |
| PolarFire | None | Its hash-verified Carnot dispatch already satisfies the continuity terminal state. |
| GateMate | None | No new operator-authored physical-state receipt exists after Exp6559. Repeating the same zero-command audit would be a doomed rerun. The next hardware action stays blocked until physical state changes. |
| Extropic TSU | None | No physical device is available. No TSU speed, power, or availability claim is permitted. |

## Milestone exit criteria

- Every task from Exp6768 through Exp6780 has a terminal artifact or an explicit capstone missing-row
  entry.
- Every comparative task emits per-unit rows and passes cold row recomputation.
- Every blocked verdict includes `gate_check_summary` with the failed check and observed value.
- Every artifact declares `verdict_class` from the closed enum.
- Every local LLM task records exact model IDs, paths, hashes, tokenizer source, device, and teardown.
- FR12 progress requires exact-valid gain, not format gain alone.
- FR11 progress requires prospective order-level benefit, nonzero admits and rejects, actual memory
  use, retention, support, poison, restart, and rollback gates.
- ARC adoption requires a cold actions-to-progress result. Transport success alone cannot enable a
  flag.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain unchanged.

## Explicitly deferred

- Learned diagnostic energy is deferred until the runtime grammar and cold localization audit show
  enough diverse exact-invalid rows. `.590` uses structural grammar conflicts for repair selection.
- The object-table fetch A/B is deferred. It already consumed hours and does not answer the current
  actions-to-progress question.
- Weight-changing continual learning remains Tier 3/4 work. `.590` tests Tier 2 external memory.
- KAN restructuring waits for a non-circular online learning signal.
- Physical TSU work waits for authenticated device access.
- New FPGA bitstream design and duplicate public-game solves remain out of scope.

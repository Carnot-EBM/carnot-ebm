# Research Roadmap V629

**Milestone:** `2026.09.629`  
**Title:** Dual-side verifier intervention, procedure-family self-learning, state-localized reasoning, and sampler execution continuity  
**Execution contract:** exactly 14 tasks, in order, from
`exp7151-v629-active-contract-preflight` through `exp7164-v629-capstone`  
**Roadmap YAML:** `research-roadmap-next.yaml`

## Executive summary

V628 completed all three tasks in its active roadmap. It did not produce a
scientific measurement. Each artifact was terminal and honest:

- Exp7148 found that an activated task cannot rely on
  `research-roadmap-next.yaml`. Activation consumes that file. The active
  `research-roadmap.yaml` is the execution-time contract.
- Exp7149 stopped before its source and cache checks. It recorded
  `preconditions_not_checked`.
- Exp7150 ran the real Qwen canary, but its original CUDA evidence reducer did
  not recognize the observed runtime receipt. The post-run repair now accepts
  all-layer offload requests, CUDA runtime markers, and positive task-owned
  GPU memory together. Focused tests pass. No experimental rerun has qualified
  that repair.

V629 closes those execution gaps before it makes a verifier claim. It then
tests a dual-side source-grounding intervention on Qwen and both mandated
Gemma models. In parallel, it resumes continuous self-learning from the valid
V627 chronological stream. It adds procedure-family memory and a cold
drift-versus-poison audit. A separate ARC cell tests state-localized prompt
invariance without claiming a game solve. The sampler branch separates exact
Rust parity from performance profiling. One bounded hardware task preserves
continuity on KV260, PolarFire, and GateMate.

This milestone does not reopen the retired generated-text scorer, PWA-KAN
lineages, adapter-withheld ARC LOO solves, or simulated hardware headlines.

## What V628 proved

### Positive operational evidence

1. The active roadmap is the only reliable execution-time YAML input after
   activation. A preflight that reads the consumed next-roadmap path is
   structurally wrong.
2. The real Qwen canary can reach the native server and produce task-owned
   runtime logs. The old reducer, not the version string, caused the final
   block. The repaired reducer has focused test coverage.
3. The task-progress contract worked well enough for all three tasks to write
   terminal artifacts. None stayed silent until the 1,201-second kill.

### What V628 did not prove

1. It did not prove that the repaired CUDA evidence contract passes on a new
   real model load.
2. It did not compare direct, self-check, symbolic, and dual-side verifier
   interventions.
3. It did not run prospective continuous self-learning.
4. It did not test state-localized ARC reasoning, Rust sampler parity, host
   sampler performance, or attached-board continuity.

### Reusable evidence from V626 and V627

- Exp7138 provides a valid 72-row source-grounded relational fixture.
- Exp7141 provides an immutable 108-event chronological self-learning stream.
- Exp7133 provides a corrected multiscale proposal with exact finite-law
  evidence in Python.
- Exp7134 provides bounded host evidence for the corrected Python sampler.
- Exp7130 shows that verifier-committed routing can run on three constraint
  families with exact external labels.
- Exp7106 provides earlier deterministic evidence that delayed procedural
  memory can help. It does not substitute for model-facing learning.

## The three largest gaps

### Gap 1: exact verification exists, but live extraction and intervention are unproven

Carnot can check formal constraints after they exist. It still lacks current
evidence that a flagship local model can extract a useful formal structure and
benefit from it. V629 first qualifies the repaired runtime. It then checks the
generated structure against the source and the generated answer against exact
query results. The comparison uses frozen rows and independent labels.

### Gap 2: continuous self-learning has evidence artifacts, but no completed prospective model loop

Carnot has a chronological event stream and several memory prototypes. The
latest FlowBalance task stopped after artifact initialization. V629 repairs
the transaction boundary, then runs a prospective four-arm model study. The
new arm stores de-instantiated procedure families instead of instance answers.
The cold audit separates valid drift, stale memory, and poisoned updates.

### Gap 3: corrected sampling stops before a portable execution and hardware boundary

The corrected Python sampler has exact and bounded benchmark evidence. The
Rust port never ran. The attached FPGA boards also have uneven evidence:
KV260 and PolarFire are reachable by SSH, while GateMate still requires a new
operator-authored physical-state receipt before another JTAG action. V629
requires exact Rust parity before performance profiling. It records each board
independently and makes no FPGA, ASIC, TSU, power, or speed claim without a
task-owned physical receipt.

## 2025-2026 research that changes this milestone

The full dated sweep is in `research-references.md` under **V629 Planner
Refresh**.

- OptiVer checks both generated model structure and generated solutions. V629
  maps this to a source-side relation check plus an exact solution-side SQL
  check. OpenReview: https://openreview.net/forum?id=w696Vhv5B2.
- SkillGLoW groups execution lessons by procedure and admits a new prior only
  after a no-degradation execution check. V629 tests this memory unit on the
  immutable stream. arXiv: https://arxiv.org/abs/2609.02217.
- Trace2Tower builds canonical events and multi-level skills from transition
  and outcome evidence. V629 tests a bounded deterministic projection. It does
  not claim to reproduce the paper's spectral method. arXiv:
  https://arxiv.org/abs/2609.05261.
- CAPTURE separates real preference drift from stale context and memory
  poisoning. V629 maps this to exact drift, stale-memory, and poisoned-memory
  controls. arXiv: https://arxiv.org/abs/2609.02265.
- The transformer backtracking study motivates state localization and context
  clearing for history-entangled reasoning. V629 tests same-state invariance
  on stored live ARC traces. arXiv: https://arxiv.org/abs/2605.22221.
- The revised FPGA-ASIC Ising paper and Extropic's Z1T report both separate
  host orchestration, graph placement, transfer, and sampling costs. V629 uses
  those categories in a host-only profile. arXiv:
  https://arxiv.org/abs/2602.15985 and Extropic:
  https://extropic.ai/writing/z1t.

Two findings stay on watch. Hidden-state probe work lacks a validated GGUF
activation receipt. Probabilistic model checking lacks a validated local
token-probability and PRISM path. Neither enters the V629 critical path.

## Target architecture

```text
                         exact external labels
                                  |
source fixture --> runtime qualification --> Qwen dual-side pilot
                                              |
                                              v
                                   two-Gemma replication
                                              |
                                              v
                                    independent row audit
                                              |
                                              +-------> verified intervention

chronological stream --> transaction preflight --> procedure-family memory
                                                      |
                                                      v
                                          cold drift/poison audit
                                                      |
                                                      +-------> FR-11 evidence

stored live ARC traces --> state-localized prompt views --> same-state audit
                                      |
                                      +-------> reasoning diagnostic only

Python corrected sampler --> Rust exact parity --> host orchestration profile
            |                                           |
            +------------------- placement boundaries --+
                                                        |
KV260 SSH -----------+                                  v
PolarFire SSH -------+--> independent board rows --> hardware boundary
GateMate receipt ----+

all terminal artifacts ---------------------------------------> V629 capstone
```

The exact checker remains outside the model. Model-written relations, SQL, and
answers are untrusted candidates. The self-learning controller opens feedback
only after each action. The ARC task uses stored live-path traces and does not
use game source, offline BFS, or a per-game adapter. Hardware rows never stand
in for a software parity result.

## Phase A: contract and execution substrate

### Exp7151 — V629 active Markdown and YAML task-contract preflight

Parse the design document and the active `research-roadmap.yaml` independently.
Compare all 14 rows, titles, deliverables, structured gates, producer fields,
failure histories, model rules, routing, progress duties, and prompt tails.
This task explicitly does not read `research-roadmap-next.yaml` after
activation. It is advisory. No science task gates on it.

**Deliverable:** `results/experiment_7151_v629_contract_preflight.json`

### Exp7152 — V629 execution-time source and SOTA cache delta

Recheck the selected primary sources and the exact local cache identity for all
three mandated GGUF repositories. Download nothing. Append one idempotent
execution delta to `research-references.md`. A source page is a receipt, not an
execution oracle.

**Deliverable:** `results/experiment_7152_v629_source_delta.json`

## Phase B: live dual-side verifier intervention

### Exp7153 — Post-fix source-grounding runtime qualification

Rerun one real bounded Qwen canary with the repaired evidence reducer. Require
the model hash, all-layer offload request, native CUDA runtime markers, positive
task-owned GPU memory, raw output, and timing in the same receipt. Recheck typed
label blinding and freeze a 24-row schedule. This task makes no verifier-value
claim.

**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`  
**Deliverable:** `results/experiment_7153_v629_grounding_runtime.json`

### Exp7154 — Qwen dual-side source-grounding pilot

Run four arms on the same frozen 24 rows: direct answer, self-check, relational
SQL, and dual-side verification. The dual-side arm checks whether the extracted
relation structure matches the source before it trusts exact query execution.
Report detection, accepted accuracy, harmful flips, abstention, latency, and
tokens per row and source family.

**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`  
**Gate:** `exp7153-grounding-runtime-postfix-qualification.grounding_runtime_ready_score == 1`  
**Deliverable:** `results/experiment_7154_v629_qwen_dual_side_grounding.json`

### Exp7155 — Two-Gemma dual-side grounding replication

Repeat the frozen comparison with the dense and MoE Gemma models. Do not pool
away a model or source-family reversal. This is a replication task, not a new
prompt search.

**Models:** `unsloth/gemma-4-31B-it-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`  
**Gate:** `exp7154-qwen-dual-side-grounding-pilot.qwen_dual_side_pilot_complete_score == 1`  
**Deliverable:** `results/experiment_7155_v629_gemma_dual_side_grounding.json`

### Exp7156 — Independent dual-side grounding causal audit

Start in a fresh process. Recompute every headline from raw Qwen and Gemma
rows. Check schedule identity, label blinding, source-side validity, exact
solution checks, harmful flips, abstention, and family reversals. Promote the
dual-side method only when it improves a valid control without hiding harm.

**Gate:** `exp7155-gemma-dual-side-grounding-replication.gemma_dual_side_replication_complete_score == 1`  
**Deliverable:** `results/experiment_7156_v629_grounding_causal_audit.json`

## Phase C: continuous self-learning and state-localized reasoning

### Exp7157 — FlowBalance memory transaction preflight

Repair the Exp7142 bootstrap-only failure. Prove that the 108-event stream can
drive a small transactional controller through propose, delayed feedback,
commit, reverse, and no-preference operations. The artifact must leave running
state before the first long subprocess. This task makes no learning-value
claim.

**Deliverable:** `results/experiment_7157_v629_flowbalance_transaction.json`

### Exp7158 — Prospective procedure-family continuous self-learning

Run four arms over the first 36 chronological events: no memory, raw instance
memory, verifier-balanced procedural memory, and procedure-family tower
memory. Every arm acts before feedback opens. The controller commits only
after exact delayed evidence and rejects updates with no verified preference.
The procedure-family arm stores action templates, procedures, and compact
strategies without answer text. GGUF weights remain frozen.

**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`  
**Gate:** `exp7157-flowbalance-transaction-preflight.flowbalance_transaction_ready_score == 1`  
**Deliverable:** `results/experiment_7158_v629_procedure_family_csl.json`

### Exp7159 — Frozen memory drift, poison, and transfer audit

Load the accepted Exp7158 memories in a new process. Permit no writes. Compare
no memory, accepted procedure memory, valid-drift updates, stale memory, and
poisoned-memory controls on 24 later events. Report future success, harmful
transfer, poison acceptance, protected retention, latency, and tokens per
event.

**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`  
**Gate:** `exp7158-procedure-family-memory-csl.procedure_family_csl_complete_score == 1`  
**Deliverable:** `results/experiment_7159_v629_memory_drift_poison_audit.json`

### Exp7160 — State-localized ARC same-state invariance study

Build paired prompt views from stored live ARC traces. Each pair has the same
current observation and legal actions but different prior histories. Compare
full cumulative history, cleared history, and canonical current-state views
with one frozen Qwen model. Measure action validity and same-state response
invariance. The task claims no game-level solve. It must still precheck the
live solve registry and report that no duplicate solve target was selected.

**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`  
**Deliverable:** `results/experiment_7160_v629_arc_state_invariance.json`

## Phase D: Rust execution, hardware continuity, and reconciliation

### Exp7161 — Rust corrected multiscale sampler exact parity

Port the corrected Exp7133 proposal to `carnot-samplers`. Feed Python and Rust
the same fixtures, seeds, and random draws. Compare proposals, accept decisions,
chains, energies, normalization, detailed balance, stationarity, and
reproducibility. Make no speed claim in this task.

**Deliverable:** `results/experiment_7161_v629_rust_multiscale_parity.json`

### Exp7162 — Rust multiscale sampler orchestration profile

After exact parity, benchmark Python and Rust at matched seeds, update counts,
and exact-energy budgets. Separate proposal construction, graph traversal,
energy delta, acceptance, random-number, and serialization time. Report host
ESS per second with per-seed rows and confidence intervals. Make no FPGA,
ASIC, TSU, Z1, energy, or power claim.

**Gate:** `exp7161-rust-multiscale-exact-parity.rust_multiscale_exact_parity_score == 1`  
**Deliverable:** `results/experiment_7162_v629_rust_orchestration_profile.json`

### Exp7163 — KV260, PolarFire, and GateMate bounded continuity

Record each attached board in a separate row. Use SSH for KV260 and PolarFire.
Run a bounded hash-verified board-local correctness smoke when each board is
reachable. Never use a host block-device check for KV260. Run no GateMate JTAG
command unless a new operator-authored physical-state receipt exists after
Exp7146. A missing board is a per-board block, not evidence from another board.

**Deliverable:** `results/experiment_7163_v629_three_board_continuity.json`

### Exp7164 — V629 independent evidence matrix and branch disposition

Read all 13 upstream artifacts without a structured scientific gate. Preserve
missing, blocked, null, disqualified, circular, and positive evidence. Recompute
the milestone claims from raw rows. Update `research-complete.yaml`, relevant
OpenSpec capability status, `_bmad/traceability.md`, `ops/status.md`, and
`ops/changelog.md` only with artifact-backed statements.

**Deliverable:** `results/experiment_7164_v629_capstone.json`

## Exact task contract

The execution contract contains exactly 14 tasks. The order below is binding.
The YAML must contain these full IDs, titles, deliverables, and gates.

| Order | Task ID | Exact title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7151-v629-active-contract-preflight` | V629 active Markdown and YAML task-contract preflight | `results/experiment_7151_v629_contract_preflight.json` | none |
| 2 | `exp7152-v629-source-and-cache-delta` | V629 execution-time source and SOTA cache delta | `results/experiment_7152_v629_source_delta.json` | none |
| 3 | `exp7153-grounding-runtime-postfix-qualification` | Post-fix source-grounding runtime qualification | `results/experiment_7153_v629_grounding_runtime.json` | none |
| 4 | `exp7154-qwen-dual-side-grounding-pilot` | Qwen dual-side source-grounding pilot | `results/experiment_7154_v629_qwen_dual_side_grounding.json` | `exp7153-grounding-runtime-postfix-qualification.grounding_runtime_ready_score == 1` |
| 5 | `exp7155-gemma-dual-side-grounding-replication` | Two-Gemma dual-side grounding replication | `results/experiment_7155_v629_gemma_dual_side_grounding.json` | `exp7154-qwen-dual-side-grounding-pilot.qwen_dual_side_pilot_complete_score == 1` |
| 6 | `exp7156-dual-side-grounding-causal-audit` | Independent dual-side grounding causal audit | `results/experiment_7156_v629_grounding_causal_audit.json` | `exp7155-gemma-dual-side-grounding-replication.gemma_dual_side_replication_complete_score == 1` |
| 7 | `exp7157-flowbalance-transaction-preflight` | FlowBalance memory transaction preflight | `results/experiment_7157_v629_flowbalance_transaction.json` | none |
| 8 | `exp7158-procedure-family-memory-csl` | Prospective procedure-family continuous self-learning | `results/experiment_7158_v629_procedure_family_csl.json` | `exp7157-flowbalance-transaction-preflight.flowbalance_transaction_ready_score == 1` |
| 9 | `exp7159-memory-drift-poison-cold-audit` | Frozen memory drift, poison, and transfer audit | `results/experiment_7159_v629_memory_drift_poison_audit.json` | `exp7158-procedure-family-memory-csl.procedure_family_csl_complete_score == 1` |
| 10 | `exp7160-state-localized-arc-invariance` | State-localized ARC same-state invariance study | `results/experiment_7160_v629_arc_state_invariance.json` | none |
| 11 | `exp7161-rust-multiscale-exact-parity` | Rust corrected multiscale sampler exact parity | `results/experiment_7161_v629_rust_multiscale_parity.json` | none |
| 12 | `exp7162-rust-multiscale-orchestration-profile` | Rust multiscale sampler orchestration profile | `results/experiment_7162_v629_rust_orchestration_profile.json` | `exp7161-rust-multiscale-exact-parity.rust_multiscale_exact_parity_score == 1` |
| 13 | `exp7163-three-board-continuity` | KV260, PolarFire, and GateMate bounded continuity | `results/experiment_7163_v629_three_board_continuity.json` | none |
| 14 | `exp7164-v629-capstone` | V629 independent evidence matrix and branch disposition | `results/experiment_7164_v629_capstone.json` | none |

## Dependency graph

```text
exp7151 contract preflight                         [advisory]
exp7152 source/cache delta                         [independent]

exp7153 runtime qualification
  `-- grounding_runtime_ready_score == 1 --> exp7154 Qwen pilot
        `-- qwen_dual_side_pilot_complete_score == 1 --> exp7155 Gemma replication
              `-- gemma_dual_side_replication_complete_score == 1 --> exp7156 audit

exp7157 transaction preflight
  `-- flowbalance_transaction_ready_score == 1 --> exp7158 procedure-family CSL
        `-- procedure_family_csl_complete_score == 1 --> exp7159 cold audit

exp7160 ARC state invariance                       [independent]

exp7161 Rust exact parity
  `-- rust_multiscale_exact_parity_score == 1 --> exp7162 host profile

exp7163 three-board continuity                     [independent]

all terminal artifacts -------------------------------------> exp7164 capstone
```

The capstone has no `gated_on` field. It must preserve partial milestone
evidence when an upstream branch blocks. A gate-blocked downstream task uses
`verdict_class=blocked`, not `partial`, and names the upstream field in
`gate_check_summary`.

## Model and runtime requirements

Every task that runs an experimental LLM declares an exact `MODEL_SPECS` row.
The task resolves a cached Q4 file with `cached_sota_pair()`, uses the GGUF
embedded tokenizer and chat template, and downloads nothing.

| Task | Required experimental model |
|---|---|
| Exp7153 | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Exp7154 | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Exp7155 | `unsloth/gemma-4-31B-it-GGUF`; `unsloth/gemma-4-26B-A4B-it-GGUF` |
| Exp7158 | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Exp7159 | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Exp7160 | `unsloth/Qwen3.6-35B-A3B-GGUF` |

Legacy small models may appear only in focused CPU smoke tests. They may not
provide a headline row.

## Hardware requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| Two RTX 3090 GPUs | Exp7153, Exp7154, Exp7155, Exp7158, Exp7159, Exp7160 | Load one cached Q4 model at a time. Require task-owned model hash, runtime log, placement, and GPU memory receipts. CPU smoke output cannot replace a model row. |
| Local CPU and Rust toolchain | Exp7151, Exp7152, Exp7156, Exp7157, Exp7161, Exp7162, Exp7164 | Run focused Python tests and the `carnot-samplers` Rust tests. Keep exact parity separate from timing. |
| KV260 at `ssh kria` | Exp7163 | Use SSH reachability and a bounded hash-verified board-local smoke. Never inspect a host block device as a board precondition. |
| PolarFire at `ssh polarfire` | Exp7163 | Use SSH reachability and a bounded hash-verified board-local smoke. No speed claim without a task-owned board transcript. |
| GateMate with DirtyJTAG | Exp7163 | Run a JTAG command only after a new operator-authored physical-state receipt after Exp7146. Otherwise record a per-board block and stop that branch. |
| Extropic Z1 or other TSU | none | No authenticated device exists. Z1T supplies architecture context only. |

## Prior-failure disposition

| V629 task | Prior failure | What changed |
|---|---|---|
| Exp7151 | Exp7148 `blocked_v628_contract_preflight_prerequisite_missing` | Read the active roadmap after activation. Do not require the consumed next-roadmap file. |
| Exp7152 | Exp7149 `blocked_v628_source_delta_precondition` | Check local prerequisites and leave running state before source access. |
| Exp7153 | Exp7150 `blocked_real_qwen_canary`; Exp7139 `blocked_native_llama_server` | Use the repaired CUDA reducer and the exact runtime receipt contract already covered by focused tests. |
| Exp7157 | Exp7142 `blocked_initial_schema_written_before_checks` | Test a bounded transaction lifecycle before the learning run. |
| Exp7161 | Exp7145 `blocked_no_run_pending_preconditions` | Split parity from performance and run only focused toolchain checks before the parity receipt. |
| Exp7162 | Exp7145 `blocked_no_run_pending_preconditions` | Gate the profile on a separate exact-parity artifact. |
| Exp7163 | Exp7146 `blocked_no_new_operator_physical_state_receipt_after_exp6559` | Preserve the GateMate one-action stop and add independent KV260 and PolarFire rows. |

Exp7160 is a new state-localized prompt study. It is not the disqualified
adapter-withheld LOO solve. Exp7163 and Exp7164 carry the standing 2026-05-29
operator override for active hardware continuity and routine capstones.

## Acceptance and interpretation rules

1. Completion is separate from a positive result. Each artifact exposes a
   closed `verdict_class` enum and a free-text `honest_verdict`.
2. Every comparison emits a `rows` or `per_game_results` list with one row for
   every unit. Aggregates never stand alone.
3. Every structured gate names a bare top-level field that appears in the
   upstream task's own required artifact fields.
4. A blocked artifact names the failed check, expected value, and observed
   value in `gate_check_summary`.
5. A generated verifier is never its own oracle. Exact fixture labels, exact
   SQL execution, delayed task outcomes, and finite-law enumeration remain
   outside the candidate model.
6. Exp7160 makes no game-level solve claim. If an incidental solve appears, it
   is not a headline and must carry `solve_provenance`. Only
   `live_agent_self_discovery` can support future live-path credit.
7. Every task prints a flushed progress line at each numbered phase boundary,
   before and after any model load, generation, benchmark, or subprocess, and
   at least every 300 seconds inside a long loop. No stdout gap may reach 600
   seconds.
8. A positive hardware claim requires a task-owned physical transcript.
   Host-only profiles use `execution_venue=host` and cannot imply device speed
   or energy.

## Main risks and controls

| Risk | Control |
|---|---|
| Repaired CUDA reducer still rejects the real runtime | Exp7153 is a bounded qualification gate. It records the raw log and memory receipt before any larger comparison. |
| A gate field drifts from its producer | Exp7151 compares each structured gate with the upstream required artifact fields. The task IDs and fields are literal in both files. |
| Dual-side checking merely adds calls | Use the same rows, exact labels, call opportunities, and per-family harmful-flip accounting. Exp7156 recomputes raw rows independently. |
| Memory stores answers or reads future feedback | Freeze chronological order and prompt hashes. Strip outcome text from memory. Open feedback only after each action. Hash the accepted snapshot. |
| Procedure-family memory hides negative transfer | Exp7159 uses later events, valid-drift, stale-memory, poison, and no-memory controls in a new process with writes disabled. |
| ARC repeats a proxy solve | Exp7160 prechecks the registry, uses stored live traces, and reports only same-state invariance. It has no target level. |
| Rust speed hides semantic drift | Exp7162 cannot dispatch unless Exp7161 emits exact parity score 1. |
| Board unavailability cascade-blocks software work | Exp7163 is independent and records each board separately. The capstone is ungated. |
| Long work dies after a silent interval | Every task has an explicit progress heartbeat inside loops and around all minute-scale calls. |

## Paths not reopened

- External generated-text energy scoring and Phase D scorer selection.
- PWA-KAN, within-chain KAN adaptation, and KAN-as-verifier retries.
- Adapter-withheld ARC LOO and outer-loop game solving.
- Hidden-state localizers without a frozen activation-export receipt.
- Model-written SQL, relations, scores, or critiques as ground truth.
- Offline BFS, game source inspection, or hand-built per-game adapters for ARC.
- Simulated FPGA, ASIC, TSU, or Z1 timing as a hardware result.
- Hardware speed or power claims without a task-owned physical transcript.

## Milestone completion rule

V629 completes when all 14 task IDs have terminal artifacts or conductor gate
records and Exp7164 reconciles the evidence. A blocked scientific branch does
not erase independent completed work. The milestone document and YAML remain
an exact 14-row contract throughout activation and execution.

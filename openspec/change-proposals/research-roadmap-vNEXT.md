# Carnot Research Roadmap vNEXT: Evidence, Online Constraint Memory, and Live Feasibility

**Created:** 2026-09-09
**Milestone:** `2026.09.631`
**Status:** Planned
**Supersedes:** milestone `2026.09.630`, experiments `exp7156` through `exp7158`
**Task contract:** exactly 14 tasks, `exp7159` through `exp7172`, in the order below
**Primary model:** `unsloth/Qwen3.8-27B-GGUF`

## What milestone 2026.09.630 proved

V630 ended with three terminal tasks. It produced two useful positive resources and one honest
external block.

| Result | Evidence | Consequence for V631 |
|---|---|---|
| The contract checker can detect a stale roadmap document. | `exp7156` independently parsed the Markdown and YAML and returned `complete_disqualified_v630_markdown_yaml_contract_mismatch`: the document declared 14 old V629 tasks while the active V630 YAML held three. | V631 publishes one exact task table and validates it before science. The preflight does not gate independent branches, so a documentation defect cannot erase the milestone. |
| The Qwen3.8 registry and cache path exist, but a clean runtime receipt does not. | `exp7157` found the pinned `Qwen3.8-27B-Q4_K_M.gguf`, then returned `blocked_idle_rtx_3090`. One orphaned `llama-server` held about 11 GiB on each RTX 3090. | Separate read-only process diagnosis from model loading. A persistent idle-GPU block is terminal `blocked`, not retryable `partial`. |
| The exact entity-evidence fixture is ready. | `exp7158` produced 648 reconstructable rows across supported and counterfactual conditions with source spans, entity identities, exact labels, sealed splits, and a frozen energy-term contract. | V631 can measure model-derived structured constraints and source sensitivity without using a model judge as the oracle. |

V630 did not produce a Qwen3.8 comparative verifier result, a prospective continuous-learning
result, a live ARC feasibility intervention, or a fixed-magnetization sampler receipt. Those are
the scientific starting points for V631.

## The three largest gaps to the PRD vision

1. **Constraint acquisition and verification value are still unproved on live SOTA output.**
   Carnot has strong exact executors and a 648-row entity-evidence fixture. It does not yet have an
   oracle-distinct result showing that Qwen3.8 can extract usable claim and evidence structures, or
   that a frozen alignment energy adds value over direct judgment and lexical controls.

2. **Continuous self-learning has no prospective noisy-feedback win.** The PRD requires the
   system to propose, evaluate, and safely merge constraint updates against immutable held-out
   truth. Prior memory branches produced useful fixtures, but the latest FlowBalance attempt
   stopped before checks. Carnot needs a chronological decision-before-feedback comparison with
   delayed, stale, and poisoned constraint values, held-future support, abstention, and rollback.

3. **Constraint guidance is not yet proven on the live agent path.** ARC remains the best local
   agentic test, but offline adapters, source reads, and already-registered solves cannot receive
   credit. The next safe intervention is smaller: use only runtime-visible state to reject invalid
   actions before energy ranking, then run a common-schedule live A/B. Hardware also remains a
   sovereignty boundary, so every attached board keeps an honest continuity receipt without a
   synthetic speed claim.

## Research inputs selected before design

The dated sweep is recorded in `research-references.md` under “V631 Planner Refresh.” It checked
arXiv, OpenReview, Hugging Face Papers, Semantic Scholar, GitHub, Extropic, and Logical
Intelligence before this task list was written.

- EAEV (`arXiv:2609.08267`) motivates paired entity and evidence extraction under exact source
  perturbations. It maps to `exp7162` and `exp7163`.
- Constrained online learning with noisy constraint values (`arXiv:2609.06921`) motivates a
  nonnegative windowed violation balance, not an impossible hard-zero promise under fixed noise.
  It maps to `exp7164` through `exp7166`.
- CLAMP (`arXiv:2609.08602`), FSNet (`arXiv:2506.00362`), and exact automaton-constrained
  decoding (`arXiv:2607.07026`) all separate feasibility from proposal quality. They motivate the
  exact admissibility layer in `exp7167` and the live A/B in `exp7168`.
- High-magnetization low-temperature sampling (`arXiv:2609.08873`) motivates pair-swap proposals
  that preserve a Hamming slice. It maps to `exp7169` and `exp7170`.
- Extropic's Z1T report supplies timing-category names for `exp7171`. Carnot has no authenticated
  Z1 device, so the task makes no Z1 runtime, energy, or parity claim.
- Spilled Energy (`arXiv:2602.18671`) was rechecked and rejected as a task. Carnot already retired
  that signal after `exp6980`; V631 does not reopen it.

## V631 architecture

```text
                                  ┌──────────────────────────────┐
                                  │ exact 14-task contract       │
                                  │ exp7159                      │
                                  └──────────────┬───────────────┘
                                                 │ advisory receipt
                                                 ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ read-only GPU lease/process  │────▶│ bounded Qwen3.8 JSON canary │
│ diagnosis                    │     │ exp7161                     │
│ exp7160                      │     └──────────────┬───────────────┘
└──────────────────────────────┘                    │ exact model identity
                                                    │ and real bounded output
                                                    ▼
                                      ┌───────────────────────────┐
exp7158 exact fixture ────────────────▶│ paired claim/evidence     │
                                      │ extraction + frozen energy│
                                      │ exp7162                   │
                                      └─────────────┬─────────────┘
                                                    │ immutable rows
                         ┌──────────────────────────┴─────────────────────┐
                         ▼                                                ▼
             ┌───────────────────────┐                       ┌──────────────────────┐
             │ independent source    │                       │ chronological noisy  │
             │ swap/leakage audit    │                       │ feedback stream      │
             │ exp7163               │                       │ exp7164              │
             └───────────────────────┘                       └──────────┬───────────┘
                                                                        ▼
                                                           ┌──────────────────────┐
                                                           │ budgeted online      │
                                                           │ constraint memory    │
                                                           │ exp7165              │
                                                           └──────────┬───────────┘
                                                                        ▼
                                                           ┌──────────────────────┐
                                                           │ cold retention,      │
                                                           │ poison, rollback     │
                                                           │ exp7166              │
                                                           └──────────────────────┘

runtime-visible ARC state ─▶ exact action mask exp7167 ─▶ live common-schedule A/B exp7168

fixed Hamming slice ─▶ pair-swap sampler + finite law exp7169 ─▶ independent audit exp7170

KV260 + PolarFire + GateMate ─▶ separate honest continuity rows exp7171

all terminal artifacts ───────────────────────────────────────▶ ungated capstone exp7172
```

The LLM remains a proposal and extraction substrate. Exact fixture labels, runtime action
admissibility, enumerated finite distributions, and independent recomputation remain the outcome
authorities. No same-model verdict is an oracle.

## Exact task contract

| Order | ID | Title | Deliverable | Structured gates |
|---|---|---|---|---|
| 1 | `exp7159-v631-exact-contract-preflight` | V631 exact Markdown and YAML task-contract preflight | `results/experiment_7159_v631_contract_preflight.json` | none |
| 2 | `exp7160-qwen38-lease-orphan-diagnosis` | Read-only Qwen3.8 lease and orphan-process diagnosis | `results/experiment_7160_v631_qwen38_lease_diagnosis.json` | none |
| 3 | `exp7161-qwen38-bounded-structured-canary` | Bounded Qwen3.8 structured-output runtime canary | `results/experiment_7161_v631_qwen38_structured_canary.json` | `exp7160-qwen38-lease-orphan-diagnosis.qwen38_runtime_preflight_ready_score == 1` |
| 4 | `exp7162-entity-evidence-alignment-ab` | Qwen3.8 counterfactual entity-evidence alignment comparison | `results/experiment_7162_v631_entity_evidence_alignment_ab.json` | `exp7161-qwen38-bounded-structured-canary.qwen38_structured_canary_ready_score == 1` |
| 5 | `exp7163-entity-evidence-causal-audit` | Independent entity-evidence intervention and leakage audit | `results/experiment_7163_v631_entity_evidence_causal_audit.json` | `exp7162-entity-evidence-alignment-ab.entity_alignment_comparison_complete_score == 1` |
| 6 | `exp7164-noisy-feedback-chronological-stream` | Immutable noisy-feedback chronological learning stream | `results/experiment_7164_v631_noisy_feedback_stream.json` | `exp7162-entity-evidence-alignment-ab.entity_alignment_comparison_complete_score == 1` |
| 7 | `exp7165-budgeted-online-constraint-memory-csl` | Budgeted online constraint-memory continuous self-learning comparison | `results/experiment_7165_v631_online_constraint_memory_csl.json` | `exp7164-noisy-feedback-chronological-stream.noisy_feedback_stream_ready_score == 1` |
| 8 | `exp7166-memory-retention-poison-rollback-audit` | Fresh-process retention, poison, and rollback audit | `results/experiment_7166_v631_memory_retention_poison_audit.json` | `exp7165-budgeted-online-constraint-memory-csl.online_constraint_memory_complete_score == 1` |
| 9 | `exp7167-live-arc-feasibility-mask-prototype` | Live ARC exact action-feasibility mask prototype | `results/experiment_7167_v631_arc_feasibility_mask.json` | none |
| 10 | `exp7168-live-arc-masked-policy-ab` | Live ARC masked-policy common-schedule A/B | `results/experiment_7168_v631_arc_masked_policy_ab.json` | `exp7161-qwen38-bounded-structured-canary.qwen38_structured_canary_ready_score == 1`; `exp7167-live-arc-feasibility-mask-prototype.arc_feasibility_mask_ready_score == 1` |
| 11 | `exp7169-fixed-magnetization-sampler-benchmark` | Fixed-magnetization swap sampler and exact finite-law benchmark | `results/experiment_7169_v631_fixed_magnetization_sampler.json` | none |
| 12 | `exp7170-fixed-magnetization-independent-audit` | Independent fixed-magnetization parity and mutation audit | `results/experiment_7170_v631_fixed_magnetization_audit.json` | `exp7169-fixed-magnetization-sampler-benchmark.fixed_magnetization_benchmark_complete_score == 1` |
| 13 | `exp7171-three-board-honest-continuity` | KV260, PolarFire, and GateMate honest continuity receipts | `results/experiment_7171_v631_three_board_continuity.json` | none |
| 14 | `exp7172-v631-independent-capstone` | V631 independent evidence matrix and branch disposition | `results/experiment_7172_v631_capstone.json` | none |

## Phase A: Contract and runtime boundary (`exp7159`-`exp7161`)

### `exp7159`: exact task-contract preflight

Parse the Markdown table and YAML independently. Compare the milestone, 14 full IDs, order,
titles, deliverables, structured gates, model declarations, substrate classes, prior-failure
blocks, progress-line contracts, and prompt tails. Test mismatch mutations. A readable mismatch is
`disqualified`; a missing prerequisite is `blocked`. This task is advisory and gates no science.

### `exp7160`: lease and orphan diagnosis

Inspect GPU UUIDs, compute PIDs, process age, parent process, command line, lease files, and the
Qwen3.8 cache and runner paths without loading a model or killing a process. Emit readiness only
when a named RTX 3090 allocation is honestly available and no unowned process conflicts. The known
orphan is an operator-state problem. The experiment records it; it does not seize stop authority.

### `exp7161`: bounded Qwen3.8 canary

Use a task-owned process group and port. Load only `unsloth/Qwen3.8-27B-GGUF`, generate a small
fixed token budget, and require one exact JSON object with a schema-valid entity/relation record.
This is `model_bounded_generation`, with a 10-second duration floor. It is not full generation and
makes no verifier-value claim.

## Phase B: Evidence-aligned verification (`exp7162`-`exp7163`)

### `exp7162`: counterfactual entity-evidence A/B

Select a sealed, family-balanced subset of V630's fixture before model output is read. Qwen3.8
extracts structured claim and evidence graphs and gives a direct supported/unsupported answer. A
frozen symbolic energy scores graph agreement. Compare direct judgment, frozen energy, lexical
overlap, and shuffled-energy controls on identical rows. Exact fixture labels alone determine
correctness. Report per-row source spans, counterfactual pair IDs, harmful flips, AUROC, accuracy,
coverage, and source-swap sensitivity. Completion and positive value are separate fields.

### `exp7163`: independent causal and leakage audit

In a fresh process, independently parse `exp7162` raw rows. Recompute every headline from rows,
swap source evidence within pairs, shuffle labels and energy assignments, remove entity/relation
terms one at a time, and scan prompts/features for sealed labels. A positive scientific class is
allowed only when the external exact-label comparison survives every audit. Otherwise report
`null`, `disqualified`, or `blocked` as observed.

## Phase C: Continuous self-learning (`exp7164`-`exp7166`)

### `exp7164`: immutable chronological noisy-feedback stream

Transform authentic `exp7162` rows into at least 96 events. Each event exposes only pre-feedback
features at decision time. Exact outcomes arrive later under fixed clean, delayed, stale, and
poison schedules. Seal chronology, split membership, source family, support identity, and all
hashes before controller evaluation. This task makes no learning claim.

### `exp7165`: budgeted online constraint memory

Run three equal-input arms: static no-learning, raw immediate updates, and a nonnegative
window-budget controller. Each arm must decide accept, abstain, or reject before the current exact
outcome is revealed. Only later exact outcomes may update external constraint weights. Measure
held-future utility, worst-window violation, recoverable support, update count, rollback count, and
a regret proxy for each event and schedule. This is the milestone's required continuous
self-learning experiment. Frozen Qwen3.8 weights keep the learning mechanism in auditable external
memory with a direct Rust/FPGA-oriented weight-table path.

### `exp7166`: cold retention, poison, and rollback audit

Load the saved memory state in a fresh process. Recompute the chronological run from raw events,
inject label, order, and source-family poison, and verify that support or violation regressions
trigger rollback to the exact prior state hash. Distinguish durable benefit from same-process
state, calibration leakage, and a no-op controller.

## Phase D: Live feasibility, sampling, hardware, and close (`exp7167`-`exp7172`)

### `exp7167`: live ARC action-feasibility prototype

Add a game-blind exact mask at the canonical `E3AgentPolicy`/`make_carnot_agent` action seam. It
may read only runtime-visible observations and public action schemas. It may not read game source,
offline adapters, solve traces, registry solutions, or hand-written game rules. Test valid-action
preservation and invalid-action rejection on generic synthetic action/state fixtures. This task
claims no game or level solve.

### `exp7168`: live common-schedule ARC A/B

Run masked and unmasked live agent arms with the same Qwen3.8 model, game-blind target schedule,
seeds, token budget, and attempt budget. Registry-precheck every target and exclude already
reproduced levels from any solve accounting. Measure invalid proposal rate, executed-transition
rate, runtime rejection, observation-derived progress, and intervention activation. Do not read
source or use a per-game adapter. Do not headline a game-level solve; if one occurs incidentally,
only the live agent's own attempts and runtime reverse engineering can receive provenance credit.

### `exp7169`: fixed-magnetization sampler and benchmark

Implement an exact pair-swap proposal that preserves the requested magnetization. Validate small
finite distributions against enumeration, then compare with unconstrained Gibbs plus rejection on
matched sparse Ising instances, proposals, temperatures, seeds, and wall budgets. Report
distribution distance, energy moments, effective sample size, acceptance, and time per retained
sample per unit. This is CPU software evidence, not a hardware speed claim.

### `exp7170`: independent sampler audit

Use a separate loader and enumerator to recompute the finite law and all aggregate claims from raw
rows. Mutate one spin, one coupling, one magnetization target, one acceptance ratio, and one row
order. Each scientific mutation must be detected. Record degree-16 and pair-swap mapping costs as
future hardware requirements only.

### `exp7171`: three-board continuity

Run one bounded, precondition-gated leg for each attached board. Use SSH for KV260 and PolarFire;
never use a host `/dev/mmcblk*` probe for KV260. GateMate commands require a new operator-authored
physical-state receipt newer than the last accepted receipt. Stop each leg at its first failed
precondition and preserve that expected/observed pair. Report real command receipts and timing
categories, with no acceleration, power, sampler, or Z1 claim.

### `exp7172`: independent capstone

Read every terminal V631 artifact, including blocked and disqualified artifacts. Recompute
structured gates and task completion from source bytes. Classify each branch as supported, null,
circular, blocked, disqualified, or incomplete. The capstone is intentionally ungated so upstream
blocks cannot erase the evidence matrix.

## Dependency graph

```text
exp7159  (advisory contract; no downstream gate)

exp7160 ──qwen38_runtime_preflight_ready_score──▶ exp7161
exp7161 ──qwen38_structured_canary_ready_score─▶ exp7162 ─▶ exp7163
                                                     └────▶ exp7164 ─▶ exp7165 ─▶ exp7166

exp7167 ──arc_feasibility_mask_ready_score──────────────┐
exp7161 ──qwen38_structured_canary_ready_score──────────┴─▶ exp7168

exp7169 ──fixed_magnetization_benchmark_complete_score────▶ exp7170

exp7171  (independent hardware continuity)
exp7172  (ungated aggregation over all terminal artifacts)
```

Every gate producer is in this roadmap. Every gate field is a bare top-level field in the
producer's required artifact contract with identical spelling.

## Hardware and runtime requirements

| Tasks | Required substrate | Expected wall time | Hard boundary |
|---|---|---:|---|
| `exp7159`, `exp7160`, `exp7163`-`exp7167`, `exp7169`, `exp7170`, `exp7172` | Host CPU and local files | 5-45 min each | No LLM load; declare `aggregation`, `no_model_load`, or `cpu_exact_solver_or_simulator` as appropriate. |
| `exp7161` | One or two idle RTX 3090s and cached Qwen3.8 Q4_K_M | 10-25 min | Fixed small token budget; declare `model_bounded_generation`; minimum honest duration 10 s. |
| `exp7162` | Dual RTX 3090 preferred; cached Qwen3.8 Q4_K_M | 45-75 min | Real multi-row generation; declare `model_full_generation`; minimum honest duration 60 s. |
| `exp7168` | Dual RTX 3090 preferred; live ARC agent path | 45-75 min | Real agent generation; declare `model_full_generation`; no source, adapter, or offline solve path. |
| `exp7171` | KV260 SSH, PolarFire SSH, GateMate DirtyJTAG plus dated physical receipt | 15-45 min | Each board leg stops on its first failed precondition. `execution_venue` carries the board identity; compute class remains `no_model_load`. |

An orphaned model server currently occupies both RTX 3090s. V631 does not authorize killing it.
The operator must clear it or arm the existing stop authority before `exp7160` can produce a ready
score. If it remains, the Qwen-gated branches record one terminal block and the CPU, sampler,
hardware, and capstone tasks still run.

Every task prompt requires flushed progress at phase boundaries, before and after any model load,
generation, benchmark, or subprocess, and inside long loops. No output gap may reach 600 seconds.
That protects the measured 4800-second hard cap; it does not enlarge it.

## Acceptance and stopping rules

- Contract success means exactly 14 matching Markdown/YAML rows with IDs `exp7159` through
  `exp7172` in the order above. A mismatch is a valid `disqualified` outcome, not a positive one.
- Model success requires the literal `unsloth/Qwen3.8-27B-GGUF` model specification, real runtime
  receipts, a task-owned process group, and the correct substrate class. Legacy small models may
  appear only in CPU smoke tests and never in a headline row.
- Verification value is positive only if the frozen alignment energy beats named controls on
  exact held-out labels and survives `exp7163`. A complete null is a useful result.
- Continuous learning value is positive only if decisions precede feedback, held-future utility
  improves without exceeding the registered window violation budget or shrinking recoverable
  support, and `exp7166` proves cold retention and rollback.
- ARC value is positive only for a measured live-path intervention under identical arms and a
  knowledge firewall. No offline solve, per-game adapter, source read, or already-registered level
  can become a headline.
- Sampler value is positive only if magnetization is exact, finite-law parity passes, comparative
  rows support the aggregate, and `exp7170` independently reproduces the claims.
- A hardware block is terminal when the external state is unchanged. It uses `verdict_class:
  blocked`, names the failed check in `gate_check_summary`, and does not burn two retries as
  `partial`.
- All comparisons emit per-unit rows. Every task emits `verdict_class` from the closed enum
  `positive | circular_positive | null | blocked | disqualified | partial` next to
  `honest_verdict`.

## Non-goals

- Do not publish or push.
- Do not modify `research-roadmap.yaml` or `scripts/research_conductor.py`.
- Do not reopen Spilled Energy, retired external-text scoring, retired KAN lineages, or
  task-specific offline ARC solvers.
- Do not claim Kona, Z1, PASS, FPGA, or thermodynamic-hardware parity without authenticated device
  execution.
- Do not treat fixture construction, task completion, same-model confidence, or exact
  admissibility as proof of verifier utility.


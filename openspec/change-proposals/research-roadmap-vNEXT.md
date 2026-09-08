# Research Roadmap V628: Measured Intervention, Continuous Learning, and State-Localized Reasoning

**Milestone:** `2026.09.628`  
**Status:** Proposed  
**Contract:** Exactly 12 tasks, in order, from
`exp7148-v628-contract-preflight` through `exp7159-v628-capstone`  
**Execution file:** `research-roadmap-next.yaml`

## Executive summary

Milestone `2026.09.627` completed all 12 scheduled tasks. It proved that the
roadmap contract, a 72-row source-grounded fixture, and a 108-event
chronological self-learning stream can be built with auditable receipts. It
also showed that the milestone still produced no varying model-facing
verification or learning result. The source comparison stopped at a false
CUDA preflight. Its downstream audit was gate-blocked. The self-learning task
wrote a blocked skeleton, so its cold audit was gate-blocked. The ARC LOO cell
was disqualified because its arms did not share one clean configuration. The
Rust sampler task did not run.

V628 converts those ready inputs into bounded measurements. It repairs the
source verifier preflight, runs a Qwen pilot, and then tests both Gemma
families. It executes continuous self-learning over a sealed chronological
prefix and tests frozen memory on later events. It replaces the retired ARC
LOO branch with state-localized context. It separates exact Rust parity from
throughput and orchestration profiling.

The milestone does not reopen energy-as-generator, PWA-KAN, the external
generated-text scorer family, or adapter-withheld ARC LOO. It makes no FPGA,
ASIC, TSU, Kona, or Z1 execution claim.

## What V627 proved

V627 established these facts:

1. The Markdown and YAML task contract can agree over all 12 tasks.
2. The source-grounded fixture is ready with 72 balanced, label-sealed rows.
3. The chronological self-learning stream is ready with 108 immutable events.
4. The local cache contains all three mandated Q4 GGUF families.
5. The original symbolic comparison did not run because its native CUDA probe
   produced a false negative.
6. The original self-learning and Rust tasks did not produce measurements.
7. Adapter-withheld LOO is not a valid ARC generalization path and is retired.
8. Gate and verdict integrity prevented blocked or disqualified work from
   becoming a positive scientific claim.

These are readiness and failure-isolation results. They are not evidence that
source grounding improves detection, memory improves future behavior, ARC
generalization improved, or the Rust sampler matches Python.

## Three largest gaps to the PRD vision

### Gap 1: no useful verifier intervention on live SOTA outputs

Carnot has exact verification infrastructure and a strong FoVer detector, but
V627 did not show that a source-grounded method changes a live local model's
decision usefully. The new work must report unique catches, harmful flips,
abstention, latency, and per-model results against sealed external labels.

### Gap 2: continuous self-learning has inputs but no measured learning loop

The PRD requires autonomous directed self-learning. V627 created a valid
chronological stream but did not execute the update loop. V628 must act before
the current outcome opens, admit memory only after exact delayed feedback,
freeze that memory, and measure later-event value and negative transfer.

### Gap 3: promising prototypes do not reach general or production paths

The live ARC branch still lacks adapter-free generalization evidence. The
corrected multiscale sampler has Python evidence but no Rust implementation.
V628 addresses both translation failures with state-localized live-agent
context and exact Python/Rust parity before any performance claim.

## Research inputs

The dated source ledger is in `research-references.md` under “V628 Planner
Refresh.” Two findings change this milestone directly:

- arXiv:2605.22221 isolates scattered state retrieval and history
  entanglement in backtracking traces. V628 maps its retraining-free context
  localization idea to ARC. The local path does not reproduce selective state
  attention.
- arXiv:2602.15985, revised 2026-09-04, treats graph decomposition and
  subproblem delivery as first-class Ising-system costs. V628 profiles these
  costs on the host only after exact Rust parity.

FlowBalance, fixed-schema memory portability, source-grounded symbolic
checking, and the corrected multiscale proposal remain the controls for the
unfinished V627 branches. Recent KAN work does not reverse prior negative
results, so V628 does not reopen it.

## Target architecture

```text
                         sealed external labels
                                  |
source + candidate ---> claim/relation view ---> exact scorer
        |                       |                    |
        +--> direct arm         +--> grounded arm   +--> per-row effect
                    \                 /                      |
                     \-- same local GGUF and budget --------+

chronological event ---> frozen GGUF action ---> delayed exact outcome
       |                                             |
       +--> no memory                                +--> admitted memory
       +--> raw trace                                +--> signed strategy memory
       +--> procedural memory                        |
                                                     v
                                       frozen later-event transfer audit

live ARC observation + current state ---> state-localized context
                    |                             |
                    +--> cumulative control       +--> localized treatment
                                      same policy, model, and budget

Python corrected multiscale sampler ---> exact shared random receipts
                    |                              |
                    +--> finite-law authority      +--> Rust port
                                                     |
                                                     v
                                      host orchestration profile
```

The verifier never becomes its own oracle. Exact source labels and exact
environment outcomes remain outside the model-facing path. The self-learning
controller changes only external memory. It does not change GGUF weights.

## Phase A: contract and execution inputs

### Exp7148 — V628 Markdown and YAML task-contract preflight

Parse the two contract files independently. Compare the exact 12 tasks,
ordering, titles, deliverables, gates, upstream producer fields, model rules,
prior-failure metadata, prompt tails, focused tests, and progress obligations.
This task is advisory. No science task depends on it.

**Deliverable:** `results/experiment_7148_v628_contract_preflight.json`

### Exp7149 — V628 execution-time source and SOTA cache delta

Refresh the required 2025-2026 sources and all three local GGUF cache receipts.
Append one idempotent execution delta to `research-references.md`. Source pages
remain receipts, not execution oracles.

**Prior-failure contract:** `experiment_id` is
`exp6461-v556-sota-source-and-benchmark-delta`. Its verdict was
`blocked_primary_source_receipt: primary pages are source receipts only; no
product, hardware, model, or ARC execution oracle was invoked.` V628 reuses the
successful V627 receipt-only contract. It keeps sources non-oracular, checks
cache metadata separately, and permits a terminal null when no method changes
the plan. Set `retire_if_same_verdict` to `true`.

**Deliverable:** `results/experiment_7149_v628_source_delta.json`

## Phase B: source-grounded verifier intervention

### Exp7150 — Source-grounding runtime and blinding preflight repair

Repair the false native-CUDA check with binary linkage and a real bounded
Qwen canary. Replace the over-broad prompt-blinding word scan with a typed
sealed-field check. Freeze a 24-row call schedule. This task proves only that
the measured path is ready.

**Deliverable:** `results/experiment_7150_v628_grounding_preflight.json`

### Exp7151 — Bounded Qwen symbolic-grounding intervention pilot

Run direct, two-pass self-check, and two-pass source-to-relation SQL arms on
the same 24 rows with `unsloth/Qwen3.6-35B-A3B-GGUF`. Preserve prompts, raw
outputs, relations, SQL, executor receipts, labels, and per-row metrics. The
sealed external label is the scorer. The SQL proposal is untrusted.

**Deliverable:** `results/experiment_7151_v628_qwen_grounding_pilot.json`

### Exp7152 — Two-Gemma symbolic-grounding replication

Repeat the frozen 24-row schedule with
`unsloth/gemma-4-31B-it-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Report model-specific reversals and do not
pool away a family loss.

**Deliverable:** `results/experiment_7152_v628_gemma_grounding_replication.json`

## Phase C: continuous self-learning

### Exp7153 — Bounded chronological verifier-balanced memory learning

Run the existing four-arm memory controller over the first 36 sealed events.
All arms act before the current outcome opens. Admit, reverse, or reject
strategy memory only after the delayed exact result. Write an immutable memory
snapshot after four complete episodes. GGUF weights stay unchanged.

**Deliverable:** `results/experiment_7153_v628_flowbalance_memory_prefix.json`

### Exp7154 — Frozen-memory later-event transfer audit

Load the immutable Exp7153 memory in a fresh process. Compare it with no memory
on 24 later chronological events. Permit no writes during evaluation. Report
future success, protected retention, stale-memory harm, negative transfer,
latency, and token cost per event.

**Deliverable:** `results/experiment_7154_v628_memory_future_transfer.json`

## Phase D: state localization, Rust parity, and reconciliation

### Exp7155 — State-localized ARC trace and same-state invariance fixture

Create an adapter-free prompt transform from stored live ARC traces. Keep the
current observation, legal action schema, and compact state summary. Remove
irrelevant trajectory text. Build same-state pairs with different histories
and require identical localized model views. This is a fixture result, not a
game solve.

**Deliverable:** `results/experiment_7155_v628_state_localized_arc_fixture.json`

### Exp7156 — Paired live ARC state-localized context comparison

Compare cumulative and state-localized contexts with the same live policy,
Qwen model, action budget, generation budget, and eligible game set. Use no
target adapter, game source, offline BFS, or per-game calibration. Any level
solve must carry `solve_provenance=live_agent_self_discovery`.

**Deliverable:** `results/experiment_7156_v628_live_arc_context_ab.json`

### Exp7157 — Rust multiscale sampler exact parity

Port the corrected Exp7133 proposal to `carnot-samplers`. Feed Python and Rust
the same fixtures, seeds, and random draws. Compare proposals, accept
decisions, chains, energies, normalization, detailed balance, stationarity,
and reproducibility. Make no speed claim in this task.

**Deliverable:** `results/experiment_7157_v628_rust_multiscale_parity.json`

### Exp7158 — Rust multiscale throughput and orchestration profile

After exact parity, benchmark Python and Rust at matched seeds, update counts,
and exact-energy budgets. Separate proposal construction, graph traversal,
energy delta, acceptance, random-number, and serialization time. Report host
ESS per second and confidence intervals. Make no FPGA, ASIC, TSU, or Z1 claim.

**Deliverable:** `results/experiment_7158_v628_rust_orchestration_profile.json`

### Exp7159 — V628 capstone and evidence reconciliation

Read all 11 upstream artifacts without a scientific gate. Exclude flagged
artifacts. Separate completion, positive evidence, null evidence, blocked
work, disqualification, and circular evidence. Reconcile the research ledger,
OpenSpec status, `_bmad/traceability.md`, `ops/status.md`, and
`ops/changelog.md` with the actual artifacts.

**Deliverable:** `results/experiment_7159_v628_capstone.json`

## Exact task contract

The execution contract contains exactly 12 tasks. The order below is binding.

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7148-v628-contract-preflight` | V628 Markdown and YAML task-contract preflight | `results/experiment_7148_v628_contract_preflight.json` | none |
| 2 | `exp7149-v628-source-and-cache-delta` | V628 execution-time source and SOTA cache delta | `results/experiment_7149_v628_source_delta.json` | none |
| 3 | `exp7150-source-grounding-preflight-repair` | Source-grounding runtime and blinding preflight repair | `results/experiment_7150_v628_grounding_preflight.json` | none |
| 4 | `exp7151-qwen-symbolic-grounding-pilot` | Bounded Qwen symbolic-grounding intervention pilot | `results/experiment_7151_v628_qwen_grounding_pilot.json` | `exp7150-source-grounding-preflight-repair.grounding_preflight_ready_score == 1` |
| 5 | `exp7152-gemma-symbolic-grounding-replication` | Two-Gemma symbolic-grounding replication | `results/experiment_7152_v628_gemma_grounding_replication.json` | `exp7151-qwen-symbolic-grounding-pilot.qwen_grounding_pilot_complete_score == 1` |
| 6 | `exp7153-flowbalance-memory-prefix-learning` | Bounded chronological verifier-balanced memory learning | `results/experiment_7153_v628_flowbalance_memory_prefix.json` | none |
| 7 | `exp7154-frozen-memory-future-transfer` | Frozen-memory later-event transfer audit | `results/experiment_7154_v628_memory_future_transfer.json` | `exp7153-flowbalance-memory-prefix-learning.flowbalance_micro_csl_complete_score == 1` |
| 8 | `exp7155-state-localized-arc-fixture` | State-localized ARC trace and same-state invariance fixture | `results/experiment_7155_v628_state_localized_arc_fixture.json` | none |
| 9 | `exp7156-live-arc-context-localization-ab` | Paired live ARC state-localized context comparison | `results/experiment_7156_v628_live_arc_context_ab.json` | `exp7155-state-localized-arc-fixture.state_localized_arc_fixture_ready_score == 1` |
| 10 | `exp7157-rust-multiscale-exact-parity` | Rust multiscale sampler exact parity | `results/experiment_7157_v628_rust_multiscale_parity.json` | none |
| 11 | `exp7158-rust-multiscale-orchestration-profile` | Rust multiscale throughput and orchestration profile | `results/experiment_7158_v628_rust_orchestration_profile.json` | `exp7157-rust-multiscale-exact-parity.rust_multiscale_parity_score == 1` |
| 12 | `exp7159-v628-capstone` | V628 capstone and evidence reconciliation | `results/experiment_7159_v628_capstone.json` | none |

The IDs, titles, deliverables, order, and gates in
`research-roadmap-next.yaml` must match this table exactly.

## Dependency graph

```text
exp7148 contract preflight                   exp7149 source/cache delta

exp7150 grounding preflight
   `-- grounding_preflight_ready_score == 1 --> exp7151 Qwen pilot
          `-- qwen_grounding_pilot_complete_score == 1 --> exp7152 Gemma replication

exp7153 chronological memory prefix
   `-- flowbalance_micro_csl_complete_score == 1 --> exp7154 future transfer

exp7155 state-localized ARC fixture
   `-- state_localized_arc_fixture_ready_score == 1 --> exp7156 live ARC A/B

exp7157 Rust exact parity
   `-- rust_multiscale_parity_score == 1 --> exp7158 orchestration profile

all terminal artifacts --------------------------------------------> exp7159 capstone
```

The capstone is ungated. It must record missing or blocked upstreams once. It
must not use `partial` for an external or terminal upstream condition.

## Acceptance and claim gates

### Contract gate

- Exactly 12 task rows agree between Markdown and YAML.
- Every gate names an earlier task and a bare top-level required artifact
  field with identical spelling.
- Every comparative task declares and emits per-unit rows.
- Every task declares the closed `verdict_class` enum.
- Every blocked artifact includes `gate_check_summary`.

### Source-grounding gate

- Exp7150 proves the actual native server can load and generate with Qwen.
- The 24-row schedule is label blind and byte frozen.
- Exp7151 completes all declared Qwen arms before any uplift claim.
- Exp7152 completes both Gemma families and reports each family separately.
- Positive intervention evidence requires a positive paired lower bound,
  useful unique catches, and no hidden-label exposure.
- A complete zero or harmful result is terminal null, not partial.

### Continuous self-learning gate

- Exp7153 processes 36 chronological events in order.
- Every action receipt predates the current exact outcome receipt.
- Updates occur only at sealed episode boundaries.
- Ties and no-preference groups produce no write.
- Model-weight hashes remain unchanged.
- Exp7154 loads memory in a fresh process and permits no evaluation writes.
- Positive future value requires a positive paired lower bound and no loss on
  the protected retention set.

### ARC gate

- Exp7155 creates same-state, different-history pairs without target adapters.
- The localized view is invariant for every valid same-state pair.
- Exp7156 uses one common model, policy, action budget, token budget, and game
  set for both arms.
- No game source, exhaustive offline solver, hand GameAdapter, or per-game
  calibration enters the live path.
- Any game-level solve declares `solve_provenance`.
- A positive claim requires live-agent self-discovery and a paired improvement
  with a non-degenerate control.

### Rust and hardware gate

- Exp7157 must establish exact shared-random parity before Exp7158 runs.
- Exp7158 excludes compilation and warmup from timed rows.
- Timed rows use matched seeds, updates, and exact-energy budgets.
- Host profiling cannot become an FPGA, ASIC, TSU, Z1, Kona, or asymptotic
  speed claim.

## Hardware requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Two local RTX 3090 GPUs | Exp7150-Exp7154 and Exp7156 | Run cached Q4 GGUF inference. Use one process per leased device. Download no model. |
| Host CPU and system memory | All tasks | Run exact labels, SQLite sandboxing, chronological memory, ARC orchestration, Rust tests, and statistics. |
| Rust toolchain | Exp7157-Exp7158 | Build `carnot-samplers`, run focused tests, and profile host execution. |
| Existing local GGUF cache | Exp7149-Exp7154 and Exp7156 | Use the three mandated repositories. Record file, revision, quantization, size, hash, template, backend, and device. |
| FPGA, ASIC, XTR-0, Z1, GateMate, KV260 | none | Not required and not authorized as evidence in V628. |

Extropic Z1 and the FPGA-ASIC paper motivate future placement and
orchestration work. They do not provide local hardware. The attached GateMate
state has no new operator receipt, and V628 does not repeat the blocked JTAG
task.

## Risk controls

- Every task writes a schema-complete running artifact before preflight.
- Every prompt requires flushed progress at every phase boundary, around every
  slow call, and inside long loops. No stdout gap may reach 600 seconds.
- Each long experiment is bounded below the 4,800-second hard cap.
- Every model task uses at least one mandated SOTA GGUF model.
- All model calls preserve raw output and exact identity receipts.
- Labels remain sealed until all arm outputs for the unit are frozen.
- Exact rejects cannot be overridden by a learned or model-generated score.
- Comparative tasks emit one row per unit, arm, seed, game, or condition.
- Blocked and disqualified results are terminal. `partial` is reserved for
  incomplete local work that a retry can finish.
- The capstone skips adversarially flagged artifacts and records
  `excluded_flagged_upstreams`.

## Explicitly closed paths

V628 does not run these paths:

- energy-as-generator or token-level energy decoding;
- the retired external generated-text or log-probability scorer family;
- PWA-KAN, KAN certificate, or within-chain adaptive coupling reruns;
- adapter-withheld ARC LOO, target adapters, offline ground-truth BFS, or
  per-game calibration;
- hardware execution without a new operator-authored physical receipt;
- Kona or Z1 benchmarks without public compatible hardware and software.

## Milestone completion rule

The milestone completes when all 12 tasks have terminal artifacts. Scientific
promotion is separate from task completion. Exp7159 may promote only claims
that survive artifact validation, adversarial verification, per-row
recalculation, provenance checks, and the gates above.

# Research Roadmap vNEXT - Milestone 2026.05.291

**Title:** Live SOTA Verifier Repair + FR-11 EvoEnv + Bounded Energy Monitors
**Created:** 2026-05-26
**Status:** Planned
**Supersedes:** 2026.05.290 "Certified Verifier Recovery + Explicit Repair Boundary + FR-11 Curriculum Guard"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.290 Proved

Milestone `.290` completed every scheduled task and repaired several missing
or ambiguous evidence surfaces, but it did not make the project paper-ready.
The authoritative closeout is
`results/experiment_3121_capstone_v290.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=36`
- `blocker_delta_from_v23=0`
- `next_top_gap=publishable_verifier_repair_headline_evidence`
- `verifier_gain_status=solver_certified_ready_no_live_sota_lift`
- `repair_claim_status=bounded_micro_panel_zero_delta_no_promotion`
- `fr11_self_learning_status=controller_only_soundness_zero_completeness_zero_no_weight_update`
- `ebt_arm_status=projection_only_sidecar_correlation_no_live_model_integration`
- `sampler_hardware_status=bounded_clut_cpu_only_no_hardware_speedup`

The main positive result is that solver authority is now real rather than
aspirational. `exp3111` emitted 72 certificates over exact fixtures with
Z3/MCS-style localization and vacuity guards. `exp3112` and `exp3113` showed
diagnostic verifier gains on exact labels with `false_accept_rate=0.0`, but
the evidence did not use a full mandated SOTA cache pair. `exp3114` localized
fragment-level verification targets, and `exp3115` finally wrote the repair
micro-panel artifact, honestly reporting `repair_success_delta=0.0`.

FR-11 also improved its boundary: `exp3116` reported zero soundness and
completeness mistakes for the new controller-only unsolvable curriculum guard,
with positive prior-retention and family-holdout deltas. It still made no
weight-update claim. Architecture work remained bounded: `exp3117` measured
EBT/ARM sidecar correlation without live integration, `exp3118` integrated a
CPU-only cLUT backend with no hardware claim, and `exp3119` preserved
GateMate/SSQA blocked status due to missing operator-visible evidence.

| Area | `.290` result | `.291` consequence |
| --- | --- | --- |
| Local SOTA cache | Only `unsloth/gemma-4-26B-A4B-it-GGUF` appeared in headline-ready cache artifacts; Qwen3.6 and Gemma-4-31B stayed missing | Start with a v2 cache/precondition manifest and block headline live claims if no mandated model is usable |
| Certified verification | Solver-certified feedback exists; diagnostic gains were solver/exact-label driven, not live SOTA lift | Run a difficulty-stratified local SOTA panel and a prefix-bound verifier pilot under explicit cache/precondition accounting |
| Repair | Artifact exists but zero repair delta and no intent preservation | Try multi-turn repair only with monitors, exact tests, and no headline wording promotion |
| Multi-turn reasoning | Fragment-level verification exists, but no satisfiable-drift audit | Add interwhen/DRIFT-inspired fragment-time monitors and a returned-answer ledger consistency gate |
| FR-11 | Controller-only guard is cleaner but not continuous self-learning by PRD standards | Add EvoEnv-style verifiable environment synthesis with solve-verify asymmetry and retention gates |
| EBT/ARM | Sidecars remain projection-only | Measure auditable energy budgets over exact fixtures and local traces before any integration claim |
| KAN/KAEM | KAN verification remains architecture context | Add one bounded MILP/PWA audit to turn KAN interpretability into a verifier artifact |
| Hardware | cLUT CPU backend works; GateMate/SSQA still require operator evidence | Keep hardware as evidence ingestion and sampler-boundary accounting, not board execution |

## Three Biggest Gaps To PRD Vision

1. **Publishable verifier and repair evidence.** The PRD requires verifiable
   reasoning and useful repair under solver/test authority. Carnot now has
   certified solver feedback but still lacks a live mandated-SOTA verifier lift
   and any nonzero repair result. `.291` must separate cache/precondition
   readiness from solver readiness, stratify live verifier evidence by
   difficulty, add bounded deterministic probability/coverage checks, and only
   run repair through exact tests and monitors.

2. **Continuous self-learning beyond controller-only guards.** FR-11 asks for
   autonomous self-learning. `.290` proved a safer curriculum controller, but
   it did not synthesize reusable learning environments, update weights, or
   show durable memory beyond controller state. `.291` must test a bounded
   EvoEnv-style loop: generate or admit executable constraint environments,
   prove solve-verify asymmetry, track retention, and reject answer leakage.

3. **Architecture-to-evidence bridge.** The long-term Carnot architecture
   points to EBT/ARM energy diagnostics, KAN/KAEM verification, and sampler
   hardware. Current artifacts are projection-only or CPU-only. `.291` should
   convert those ideas into bounded evidence: energy-budget diagnostics,
   KAN PWA/MILP verification accounting, and operator-evidence-only hardware
   status.

## New Research Integrated

The post-`.290` planning sweep was appended to `research-references.md` before
this milestone was designed. Findings shaping `.291`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| EvoEnv reframes self-improvement as verifiable environment synthesis with stable solve-verify asymmetry | arXiv:2605.14392 / Hugging Face papers | `exp3128` builds the required FR-11 continuous self-learning pilot around executable, solver-labeled environments |
| ReVeal trains generation and verification through multi-turn tool feedback | OpenReview `q56ZI1Co43` | `exp3127` tries a multi-turn repair ladder with tests and Z3 as authority |
| interwhen verifies partial traces at runtime with plug-in monitors | Microsoft Research / GitHub | `exp3126` adds fragment-time monitors before final repair |
| BEAVER computes sound probability bounds for prefix-closed semantic constraints | OpenReview `xO3efBXHM9` | `exp3125` pilots prefix-closed verifier bounds on bounded exact fixtures |
| Variation in Verification warns that verifier scale and generator strength interact with difficulty | OpenReview `DcEuBwrWnB` | `exp3124` stratifies local SOTA verifier results by exact difficulty and generator family |
| DRIFT-Bench shows residual failures are often satisfiable drift after MUS repair | OpenReview `B9gtT1hhEm` | `exp3126` and `exp3129` add ledger-consistency and returned-answer drift fields |
| ARM-EBM v4 was revised 2026-05-25 and EBT has active 2026 follow-ons such as LoopUS and CEM | arXiv:2512.15605, 2507.02092, 2605.11011, 2605.07588 | `exp3130` measures energy budgets and approximation gaps rather than claiming integration |
| KAN property verification via PWA/MILP remains the most concrete KAN verifier path | arXiv:2602.06737 | `exp3131` creates a bounded KAN abstraction audit |
| FPGA/p-bit and thermodynamic hardware literature remains active but external | arXiv:2506.00269, 2512.24558, Extropic hardware page | `exp3132` preserves hardware evidence boundaries and avoids speedup claims |

## Architecture Direction

`.291` keeps exact solvers, executable tests, and certified monitors as the
only authorities. Local SOTA GGUF models may generate candidates, traces, or
probabilities, but they cannot certify their own correctness. EBT/ARM, KAN,
and sampler layers remain sidecars until they produce bounded local evidence.

```text
                +-------------------------------------+
                | .290 capstone + matrix v24          |
                | paper_ready=false, blockers=36      |
                +-------------------+-----------------+
                                    |
                                    v
        +---------------------------+---------------------------+
        | exp3122 archive + exp3123 SOTA cache/preconditions    |
        +---------------------------+---------------------------+
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
   +----------+-----------+                    +----------+-----------+
   | exp3124 live SOTA    |                    | exp3128 FR-11        |
   | verifier panel       |                    | EvoEnv synthesis     |
   +----------+-----------+                    +----------+-----------+
              |                                           |
              v                                           v
   +----------+-----------+                    +----------+-----------+
   | exp3125 prefix-bound |                    | exp3129 constraint   |
   | deterministic pilot  |                    | memory/drift audit   |
   +----------+-----------+                    +----------+-----------+
              |
              v
   +----------+-----------+
   | exp3126 fragment-time|
   | monitors + drift     |
   +----------+-----------+
              |
              v
   +----------+-----------+
   | exp3127 multi-turn   |
   | repair ladder        |
   +----------------------+

   +----------------------+      +----------------------+      +----------------------+
   | exp3130 ARM/EBT      |      | exp3131 KAN PWA/MILP |      | exp3132 hardware     |
   | energy budget        |      | verifier audit       |      | evidence boundary    |
   +----------+-----------+      +----------+-----------+      +----------+-----------+
              \                         |                         /
               \                        |                        /
                v                       v                       v
                  +---------------------+---------------------+
                  | exp3133 matrix v25 + exp3134 capstone     |
                  +-------------------------------------------+
```

## Required SOTA Model Policy

Every `.291` experiment that invokes a local LLM must include `MODEL_SPECS` and
must attempt at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE)
- `unsloth/gemma-4-31B-it-GGUF` (flagship dense)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE)

Legacy small models such as `Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear
only as CPU smoke tests. They cannot headline verifier, repair, or
self-learning results. Tasks that need live models must record cache status,
exact model IDs, selected quantization/file when known, prompt hashes, live-call
counts, GPU/precondition checks, and whether headline evidence was skipped
because no mandated model was usable.

## Milestone Phases

### Phase A - Archive and Local SOTA Readiness

**Goal:** make `.291` activation and model availability mechanical before any
headline verifier or repair run.

- `exp3122` archives `.290`, carries forward matrix v24 and capstone blockers,
  and stages `.291` without editing `research-roadmap.yaml`.
- `exp3123` writes the local SOTA cache/precondition manifest v2, including
  dual-RTX state, model IDs, cache files, `cached_sota_pair_available`, and
  headline-claim rules.

### Phase B - Live Verifier, Bounds, Monitors, and Repair

**Goal:** turn certified solver feedback into publishable live-model evidence,
or produce a precise blocker when cache or verifier behavior prevents that.

- `exp3124` runs a difficulty-stratified mandated-SOTA verifier panel over exact
  fixtures and reports live lift, false accepts, false rejects, and repair gate
  status.
- `exp3125` pilots a BEAVER-inspired prefix-closed deterministic verifier bound
  on small exact constraints.
- `exp3126` adds interwhen/DRIFT-style fragment-time monitors and returned
  answer ledger-consistency checks.
- `exp3127` runs a ReVeal-style multi-turn repair ladder only when the verifier
  panel reports an unblocked repair gate; tests and Z3 remain the authorities.

### Phase C - Continuous Self-Learning and Architecture Evidence

**Goal:** advance FR-11 and the EBT/KAN/sampler architecture without claiming
more than local artifacts prove.

- `exp3128` is the required continuous self-learning experiment. It builds an
  EvoEnv-style verifiable environment synthesis pilot for constraint families.
- `exp3129` audits FR-11 constraint memory, retention, and satisfiable drift
  after the environment pilot.
- `exp3130` converts ARM/EBT sidecar work into an auditable energy-budget
  diagnostic over exact fixtures and local traces.
- `exp3131` builds a bounded KAN PWA/MILP verifier abstraction audit.
- `exp3132` ingests hardware/operator evidence and updates sampler boundaries
  without flashing, synthesizing, or claiming speedup.

### Phase D - Matrix and Capstone

**Goal:** close from artifacts, not intent.

- `exp3133` builds cross-corpus matrix v25 with rows for SOTA cache coverage,
  live verifier lift, prefix bounds, drift monitors, repair, FR-11 EvoEnv,
  energy budgets, KAN verification, and hardware boundaries.
- `exp3134` writes the `.291` capstone and recommends the next milestone from
  matrix v25.

## Dependency Graph

```text
exp3122 archive
  -> exp3123 SOTA cache/preconditions
       -> exp3124 live SOTA verifier panel
            -> exp3125 prefix-bound verifier pilot
            -> exp3126 fragment-time monitor + satisfiable drift audit
                 -> exp3127 multi-turn repair ladder
       -> exp3128 FR-11 EvoEnv synthesis
            -> exp3129 FR-11 memory/drift audit
       -> exp3130 ARM/EBT energy-budget diagnostic

exp3131 KAN PWA/MILP audit
exp3132 hardware evidence/sampler boundary

exp3124, exp3125, exp3126, exp3127, exp3128, exp3129, exp3130, exp3131, exp3132
  -> exp3133 matrix v25
       -> exp3134 capstone v291
```

Structured conductor gates are used only where skipping the agent call is
useful and safe:

- `exp3124` gates on `exp3123.sota_cache_manifest_v2_ready == true`.
- `exp3125` gates on `exp3124.exact_live_sota_panel_v6_ready == true`.
- `exp3126` gates on `exp3124.exact_live_sota_panel_v6_ready == true`.
- `exp3127` gates on `exp3124.repair_gate_state == "unblocked"` and
  `exp3126.fragment_time_monitor_v1_ready == true`.
- `exp3129` gates on `exp3128.fr11_evoenv_pilot_v1_ready == true`.
- `exp3130` gates on `exp3123.sota_cache_manifest_v2_ready == true`.
- `exp3134` gates on `exp3133.matrix_v25_ready == true`.

## Hardware Requirements

`.291` uses hardware conservatively:

- **Dual RTX 3090 CUDA:** required for live mandated GGUF attempts in
  `exp3124`, `exp3127`, `exp3128`, and `exp3130` when local cache permits.
  All compute tasks must record `preconditions_checked`,
  `gpu_preflight`, `model_cache_status`, and `inference_substrate`.
- **CPU/Z3:** required for solver certificates, prefix-bound checks, monitors,
  FR-11 environment validation, KAN MILP/PWA accounting, and matrix/capstone
  aggregation.
- **GateMate/SSQA/KV260:** no board execution is scheduled. `exp3132` may only
  inspect documented operator-provided evidence and must record
  `hardware_commands_run: []` unless the operator has explicitly supplied a
  transcript from an earlier run.
- **THRML/TSU:** architecture context only; no TSU latency, energy, or hardware
  execution claim is allowed.

## Experiment List

| ID | Title | Phase | Deliverable |
| --- | --- | --- | --- |
| exp3122 | Archive .290 and activate .291 planning | A | `results/experiment_3122_archive_v290_activate_v291.json` |
| exp3123 | Local SOTA cache and precondition manifest v2 | A | `results/experiment_3123_sota_cache_preconditions_manifest_v2.json` |
| exp3124 | Difficulty-stratified live SOTA verifier panel v6 | B | `results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json` |
| exp3125 | Prefix-closed deterministic verifier bound pilot | B | `results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json` |
| exp3126 | Fragment-time monitor and satisfiable-drift audit | B | `results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json` |
| exp3127 | Multi-turn monitored repair ladder | B | `results/experiment_3127_multi_turn_monitored_repair_ladder_v1.json` |
| exp3128 | FR-11 EvoEnv verifiable environment synthesis | C | `results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json` |
| exp3129 | FR-11 constraint memory retention and drift audit | C | `results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json` |
| exp3130 | ARM/EBT energy-budget sidecar diagnostic v2 | C | `results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json` |
| exp3131 | KAN PWA/MILP verifier abstraction audit | C | `results/experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.json` |
| exp3132 | Hardware evidence and sampler boundary v5 | C | `results/experiment_3132_hardware_evidence_sampler_boundary_v5.json` |
| exp3133 | Cross-corpus matrix v25 | D | `results/experiment_3133_cross_corpus_matrix_v25.json` |
| exp3134 | Capstone v291 | D | `results/experiment_3134_capstone_v291.json` |

## Acceptance Criteria

- All tasks either complete or honestly gate/skip with artifact-visible reasons.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain unchanged.
- Every local LLM task includes the mandated SOTA GGUF model list and records
  cache/precondition status before inference.
- No experiment promotes repair unless exact tests or solvers accept the repair
  and intent/ledger preservation is measured.
- FR-11 has at least one continuous self-learning artifact based on executable
  verifiable environments, with retention and soundness gates.
- Energy, KAN, and hardware claims stay bounded to measured local artifacts.
- `research-references.md`, `ops/status.md`, `ops/changelog.md`,
  `_bmad/traceability.md`, and this roadmap can be reconciled after execution.

## Failed-Rerun Compliance

Every task whose scope intersects previous failures or bounded/blocked attempts
has a `prior_failures` block in `research-roadmap-next.yaml` with:

- `experiment_id`
- `verdict`
- `addressed_by`
- `retire_if_same_verdict: true`

No task reuses a retired experiment ID. No `requires` or `gated_on` chain points
to a retired upstream task. The repair tasks explicitly avoid the retired
unsupported repair-headline wording path; they only collect bounded evidence.

## Decentralization Implications

`.291` preserves Carnot's local-first stance. Local open GGUF models are the
only allowed headline LLM substrate; closed APIs may not be used for core
claims. Exact solvers, executable tests, and local artifacts remain portable and
auditable. External systems such as Extropic, Kona, Microsoft interwhen, and
OpenReview papers are references, not dependencies.

## Out of Scope

- Editing `research-roadmap.yaml` or `scripts/research_conductor.py`
- Pushing changes
- Public documentation edits
- Closed-model headline evidence
- TSU/Kona/hardware speedup claims without local authenticated evidence
- Board flashing, synthesis, or readback during hardware evidence ingestion
- Promoting zero-delta repair as a positive headline

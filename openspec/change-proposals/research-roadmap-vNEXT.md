# Research Roadmap vNEXT - Milestone 2026.05.289

**Title:** Verifier/Repair Recovery + MaxSAT Routing + Sidecar Boundaries
**Created:** 2026-05-26
**Status:** Planned
**Supersedes:** 2026.05.288 "Abstention-Calibrated Verifier Recovery + Exact Fixtures + FR-11 Completeness Repair"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.288 Proved

Milestone `.288` completed, but its capstone authority still reports
`paper_ready=false`. The authoritative closeout is
`results/experiment_3094_capstone_v288.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=36`
- `blocker_delta_from_v21=-6`
- `verifier_gain_status=flagged_or_gated_verifier_gain_recovery_incomplete`
- `repair_claim_status=bounded_flagged_gated_missing_verifier_gated`
- `fr11_self_learning_status=clean_controller_only_zero_mistake_budget`
- `ebt_arm_status=projection_only_sidecar_schema_no_model_integration`
- `gatemate_status=blocked_no_rerun_operator_actions_required`
- `ssqa_status=gated_skipped_host_visible_smoke_missing`

The main positive result is that FR-11 controller-only learning recovered:
`exp3090` reported zero soundness mistakes, zero completeness mistakes, a
positive family-holdout delta, and a rollback event under a controller-only
scope. The main negative result is that verifier/repair did not recover.
`exp3085` used a mandated local SOTA GGUF, but `abstention_precision=0.0`;
`exp3086` had Z3 but no formal-feedback lift; `exp3087` gate-blocked; and
`exp3089` remained missing because repair was correctly gated behind verifier
evidence. The EBT/ARM sidecar schema shipped only as a projection, and broad
test-suite execution exposed unrelated failures that prevent calling it an
integrated inference path.

| Area | `.288` result | `.289` consequence |
| --- | --- | --- |
| Verifier gain | Abstention degraded to `0.0` precision and calibration gate-blocked | Audit the evaluation protocol, then route accept/reject/abstain with a MaxSAT policy before another SOTA panel |
| Formal feedback | Z3 available, Dafny absent, no lift over solver-only | Run a Z3/test-oracle feedback v2 with explicit vacuity guards and no Dafny claim |
| Repair | Structured emitter preflight passed, repair micro-panel missing/gate-blocked | Retry repair only if verifier calibration produces positive exact-grounded delta |
| FR-11 | Controller-only continuous self-learning was clean | Stress the controller-only boundary before any stronger self-learning claim |
| EBT/ARM | Sidecar schema and replay scorer ready, but no live model integration | Add pipeline-boundary tests and keep no-weight-update/no-speedup claims explicit |
| Hardware | GateMate/SSQA still need operator-visible evidence | Ingest evidence if present; otherwise preserve no-rerun blocked status |
| Matrix/capstone | Matrix v22 reports 36 blockers, 17 verifier/repair blockers, 11 hardware blockers | Prioritize verifier/repair first, then hardware evidence, then publication matrix |

## Three Biggest Gaps To PRD Vision

1. **Exact verifier and repair evidence gap.** The PRD requires verifiable
   reasoning and repair under exact authority. Carnot still has 17
   verifier/repair blocker rows, a missing repair micro-panel artifact, and
   only negative or gated verifier evidence. `.289` must first make the exact
   fixture evaluation protocol non-tiny and auditable, then test MaxSAT-routed
   abstention/calibration with mandated local SOTA GGUFs.

2. **Self-learning promotion boundary gap.** FR-11 is now clean only in a
   controller-only loop. The PRD's long-term autonomous self-learning vision
   requires broader nonforgetting and adaptation, but `.288` did not update
   model weights or prove generalized learning. `.289` should stress the
   controller-only loop across harder families, delayed regressions, and
   negative controls while preserving strict promotion boundaries.

3. **Architecture-to-evidence bridge gap.** Carnot's architecture points toward
   EBT/ARM energy scoring and hardware-accelerated sampling, but current local
   artifacts are sidecar/projection-only and operator-evidence blocked. `.289`
   should add sidecar pipeline boundary tests and a CPU cLUT sampler microbench
   as architecture scaffolding, without turning either into a hardware or model
   integration claim.

## New Research Integrated

The post-`.288` planning sweep was appended to `research-references.md` before
this milestone was designed. Findings shaping `.289`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| MaxSAT/MaxSMT can frame LLM routing as constraint optimization over hard and soft preferences | OpenReview `Qmr9VbwRaB` | Define an auditable accept/reject/abstain routing policy before verifier calibration |
| Autoregressive generation can be analyzed with entropy-production diagnostics | arXiv:2604.07867 | Add optional thermodynamic decode telemetry to SOTA panels as diagnostic evidence |
| Compressed lookup-table random variate generation can reduce sampling overhead | OpenReview `hRY0ytSnM0` | Prototype a CPU cLUT sampler microbench and FPGA mapping notes without hardware speedup claims |
| MiniF2F-Dafny highlights auto-active verification opportunities and empty-proof baselines | arXiv:2512.10187 | Keep Dafny as a future corpus/toolchain target; use Z3/test-oracle feedback now |
| Formal annotation generation needs test-oracle vacuity guards | arXiv:2601.12845 | Require functional/test-oracle guards for formal-feedback lift claims |

## Architecture Direction

`.289` keeps the existing Carnot authority stack intact: exact solvers and
execution tests remain the source of truth, local SOTA GGUFs are only
candidates/judges under measurement, and energy/routing layers must expose
their constraints and failures.

```text
                   +-----------------------------+
                   | research-roadmap-next.yaml  |
                   | exp3095 ... exp3108         |
                   +--------------+--------------+
                                  |
                                  v
+------------------+     +--------+---------+      +-----------------------+
| .288 authorities | --> | fixture/protocol | ---> | MaxSAT routing policy |
| matrix v22       |     | audit exp3097    |      | exp3098              |
| capstone v288    |     +--------+---------+      +-----------+-----------+
+------------------+              |                            |
                                  v                            v
                         +--------+---------+        +---------+----------+
                         | SOTA abstention  | -----> | verifier calib v4 |
                         | panel exp3099    |        | exp3101           |
                         +--------+---------+        +---------+----------+
                                  |                            |
                                  v                            v
                         +--------+---------+        +---------+----------+
                         | Z3/test-oracle   | -----> | gated repair v3   |
                         | feedback exp3100 |        | exp3102           |
                         +------------------+        +--------------------+

+------------------+     +------------------+      +-----------------------+
| FR-11 clean      | --> | stress boundary  |      | sidecar / sampler /   |
| controller-only  |     | exp3103          |      | hardware evidence     |
+------------------+     +------------------+      | exp3104-exp3106       |
                                                    +-----------+-----------+
                                                                |
                                                                v
                                                    +-----------+-----------+
                                                    | matrix v23 + capstone |
                                                    | exp3107 + exp3108     |
                                                    +-----------------------+
```

## Required SOTA Model Policy

Every `.289` experiment that invokes an LLM must include `MODEL_SPECS` and
must attempt at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE)
- `unsloth/gemma-4-31B-it-GGUF` (flagship dense)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE)

Legacy small models such as `Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear
only as CPU smoke tests. They cannot headline verifier, formal-feedback, or
repair results. Tasks that need live models must record cache status, exact
model IDs, quantization/file selection if known, prompt hashes, and whether a
headline result was skipped because the mandated model cache was unavailable.

## Milestone Phases

### Phase A - Archive, Blockers, and Evaluation Protocol

**Goal:** make the `.288` failure surface mechanical before rerunning any
model-heavy work.

- `exp3095` archives `.288` and stages `.289` without editing the active
  roadmap.
- `exp3096` decomposes the 36 matrix v22 publication blockers into reducible,
  external-evidence, and bounded/projection-only categories.
- `exp3097` audits why `.288` exact fixture work collapsed to a tiny live
  SOTA panel and writes a stratified evaluation protocol for `.289`.

### Phase B - MaxSAT Routing and Verifier Recovery

**Goal:** replace ad hoc threshold gates with a policy that separates hard
constraints from soft tradeoffs, then measure local SOTA verifier behavior on
larger exact labels.

- `exp3098` defines the MaxSAT accept/reject/abstain routing policy.
- `exp3099` runs a local SOTA confidence/abstention panel v3 with mandated
  GGUF model specs and optional thermodynamic decode telemetry.
- `exp3100` runs a Z3/test-oracle formal-feedback v2 pilot with Dafny absence
  recorded honestly.
- `exp3101` runs verifier calibration v4 gated only on readiness signals, not
  on a pre-assumed positive precision result.

### Phase C - Gated Repair, FR-11 Stress, and Architecture Boundaries

**Goal:** try repair only if verifier evidence is positive, preserve the FR-11
controller-only boundary, and convert sidecar/sampler architecture ideas into
bounded artifacts.

- `exp3102` runs the structured repair micro-panel v3 only if `exp3101`
  reports positive MaxSAT-routed verifier delta.
- `exp3103` is the required continuous self-learning experiment. It stress
  tests the clean FR-11 controller-only result against harder holdouts,
  retention, delayed regression, and negative controls.
- `exp3104` tests EBT/ARM sidecar pipeline boundaries and targeted imports
  without claiming live model integration.
- `exp3105` prototypes a CPU cLUT random-variate sampler microbench with
  distribution-error checks and FPGA mapping notes.

### Phase D - Hardware Evidence, Matrix, and Capstone

**Goal:** preserve hardware truth boundaries and close the milestone with a
matrix/capstone that cannot promote missing or gated evidence.

- `exp3106` ingests GateMate/SSQA operator evidence if present and otherwise
  keeps reruns/readback blocked.
- `exp3107` builds cross-corpus matrix v23.
- `exp3108` writes the `.289` capstone and recommends the next milestone from
  matrix v23, not from planning intent.

## Dependency Graph

```text
exp3095 archive
  -> exp3096 blocker triage
  -> exp3097 exact fixture protocol
        -> exp3098 MaxSAT routing
              -> exp3099 SOTA abstention panel
              -> exp3101 verifier calibration v4
        -> exp3100 Z3/test-oracle feedback
              -> exp3101 verifier calibration v4
                    -> exp3102 gated structured repair
        -> exp3103 FR-11 stress boundary

exp3096 -> exp3104 EBT/ARM sidecar boundary
exp3096 -> exp3105 cLUT sampler microbench
exp3096 -> exp3106 GateMate/SSQA evidence ingestion

exp3095..exp3106 -> exp3107 matrix v23 -> exp3108 capstone
```

Structured conductor gates:

- `exp3099` requires `exp3097.eval_protocol_ready == true` and
  `exp3098.maxsat_policy_ready == true`.
- `exp3100` requires `exp3097.eval_protocol_ready == true`.
- `exp3101` requires `exp3099.abstention_panel_v3_ready == true` and
  `exp3100.formal_feedback_v2_ready == true`.
- `exp3102` requires `exp3101.verifier_gain_delta_with_maxsat > 0.0`.
- `exp3103` requires `exp3097.eval_protocol_ready == true`.
- `exp3108` requires `exp3107.matrix_v23_ready == true`.

## Hardware Requirements

- **Dual RTX 3090 CUDA:** required for `exp3099`, `exp3100`, `exp3101`, and
  `exp3102` if live local SOTA inference runs. Each task must use the cached
  SOTA pattern from `scripts/experiment_template.py`, record cache status, and
  skip headline claims if no mandated model is usable.
- **CPU-only:** sufficient for `exp3095`, `exp3096`, `exp3097`, `exp3098`,
  `exp3103`, `exp3104`, `exp3105`, `exp3107`, and `exp3108`.
- **GateMate / SSQA:** no board flash, no synthesis, and no hardware command is
  required by planning. `exp3106` may only ingest operator-provided evidence
  files. If evidence is missing, it must write a blocked artifact.
- **KV260 / PolarFire / AMD XDNA / Extropic TSU:** out of scope for `.289`
  execution. They may appear only as architecture context already tracked in
  the hardware wishlist.

## Experiment List

| ID | Title | Deliverable |
| --- | --- | --- |
| exp3095 | Archive .288 and activate .289 | `results/experiment_3095_archive_v288_activate_v289.json` |
| exp3096 | Publication blocker triage and retirement ledger v2 | `results/experiment_3096_publication_blocker_triage_and_retirement_ledger_v2.json` |
| exp3097 | Exact fixture evaluation protocol audit | `results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json` |
| exp3098 | MaxSAT abstention routing policy | `results/experiment_3098_maxsat_abstention_routing_policy_v1.json` |
| exp3099 | Local SOTA confidence/abstention panel v3 | `results/experiment_3099_local_sota_confidence_abstention_panel_v3.json` |
| exp3100 | Z3/test-oracle formal feedback v2 | `results/experiment_3100_z3_oracle_feedback_v2.json` |
| exp3101 | Local SOTA verifier calibration v4 | `results/experiment_3101_local_sota_verifier_calibration_v4.json` |
| exp3102 | Gated structured repair micro-panel v3 | `results/experiment_3102_gated_structured_repair_micro_panel_v3.json` |
| exp3103 | FR-11 ReSyn/KAN-CL stress and promotion boundary | `results/experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.json` |
| exp3104 | EBT/ARM sidecar pipeline boundary v2 | `results/experiment_3104_ebt_arm_sidecar_pipeline_boundary_v2.json` |
| exp3105 | cLUT random-variate sampler microbench | `results/experiment_3105_clut_random_variate_sampler_microbench_v1.json` |
| exp3106 | GateMate/SSQA operator evidence ingestion v3 | `results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json` |
| exp3107 | Cross-corpus matrix v23 | `results/experiment_3107_cross_corpus_matrix_v23.json` |
| exp3108 | Capstone .289 | `results/experiment_3108_capstone_v289.json` |

## Acceptance Criteria

`.289` succeeds if it produces terminal artifacts for all tasks and preserves
truth boundaries. It is not required to make the paper ready. A paper-ready
claim is allowed only if matrix v23 reports `publication_blocker_count=0` and
there are no missing artifacts, headline model-spec gaps, hardware-evidence
gaps, or projection-only rows supporting claimed results.

Minimum useful outcomes:

- A larger exact evaluation protocol is available and used by live SOTA tasks.
- MaxSAT routing exists as an auditable policy artifact.
- Verifier calibration v4 reports a measured delta, including negative or zero
  outcomes, without gate-blocking on an over-strict prior precision threshold.
- Repair v3 runs only if the verifier delta is positive; otherwise it
  gate-skips cheaply.
- FR-11 stress testing preserves the controller-only claim boundary.
- EBT/ARM and cLUT work produce bounded architecture artifacts, not integration
  or hardware speedup claims.
- Matrix v23 and capstone `.289` reconcile every missing, gated, bounded,
  projection-only, and blocked row.

## Failed-Rerun Compliance

Every task that reuses a prior scope includes `prior_failures:` in
`research-roadmap-next.yaml` with all four required fields:
`experiment_id`, `verdict`, `addressed_by`, and
`retire_if_same_verdict: true`. No task references a retired upstream in a
`requires:` chain. No task uses the retired KV260 host `/dev/mmcblk*`
precondition. No `.289` task revives retired WOPR/GRPO/SpecAnn/PIMI/OTV scope.

## Out Of Scope

- Editing `research-roadmap.yaml` during planning.
- Modifying `scripts/research_conductor.py`.
- Claiming SOTA repair without exact verifier evidence.
- Claiming EBT/ARM live model integration.
- Claiming GateMate, SSQA, KV260, PolarFire, XDNA, or Extropic hardware speedup.
- Treating legacy small GGUF smoke-test models as headline SOTA evidence.

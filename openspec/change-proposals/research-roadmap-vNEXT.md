# Research Roadmap vNEXT - Milestone 2026.05.290

**Title:** Certified Verifier Recovery + Explicit Repair Boundary + FR-11 Curriculum Guard
**Created:** 2026-05-26
**Status:** Planned
**Supersedes:** 2026.05.289 "Verifier/Repair Recovery + MaxSAT Routing + Sidecar Boundaries"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.289 Proved

Milestone `.289` completed every scheduled task except the intentionally
gate-blocked verifier-calibration and repair runs, but the capstone authority
still reports `paper_ready=false`. The authoritative closeout is
`results/experiment_3108_capstone_v289.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=36`
- `verifier_gain_status=model_spec_gap_or_gated_verifier_gain_recovery_incomplete`
- `repair_claim_status=blocked_gated_missing_verifier_gated_repair_not_promoted`
- `fr11_self_learning_status=clean_controller_only_soundness_zero_completeness_promotion_blocked`
- `ebt_arm_status=projection_only_sidecar_pipeline_no_model_integration`
- `sampler_hardware_status=diagnostic_only_cpu_microbench_no_hardware_speedup`
- `gatemate_status=blocked_no_rerun_operator_actions_required_no_speedup_claim`
- `ssqa_status=gated_skipped_host_visible_readback_missing`

The main positive result is that `.289` made the evidence surface much more
mechanical. `exp3097` produced a 72-case exact fixture protocol and `exp3098`
produced an auditable MaxSAT routing policy. `exp3099` ran a non-tiny local
SOTA panel over 48 exact cases with `false_accept_rate=0.0` and
`rejection_recall=1.0`, but `abstention_precision=0.0` and matrix v23 still
flagged a model-spec metadata gap. `exp3100` proved that Z3 was available and
vacuity guards passed, but `formal_feedback_v2_ready=false` because cached
SOTA-pair availability was treated as a hard readiness condition; the
solver-only baseline outperformed guided attempts. `exp3101` and `exp3102`
therefore gate-blocked, leaving the repair micro-panel artifact absent.

FR-11 remained honest: controller-only soundness was clean, but completeness
and retention failed promotion. `exp3103` reported `soundness_mistakes=0`,
`completeness_mistakes=12`, `prior_retention_delta=-0.444444`, and
`promotion_decision=blocked`. Architecture work stayed bounded:
`exp3104` shipped sidecar pipeline tests without live model integration,
`exp3105` shipped a CPU cLUT microbench with no hardware claim, and `exp3106`
kept GateMate/SSQA blocked on missing operator-visible evidence.

| Area | `.289` result | `.290` consequence |
| --- | --- | --- |
| Exact fixtures | 72 usable fixtures and 48-case SOTA panel floor | Reuse exact authority; do not spend a task rediscovering fixtures |
| Model specs | Mandated model IDs attempted, but matrix found headline metadata gap | Add a cache/model-spec corrigendum before any headline LLM result |
| Formal feedback | Z3 available, vacuity guarded, but v2 blocked by cached SOTA-pair condition and negative guided delta | Decouple solver-certified coherence feedback from LLM cache availability |
| Calibration | Gate-blocked on `formal_feedback_v2_ready=false` | Run diagnostic calibration gated on readiness, not on assumed positive lift |
| Repair | Missing/gate-blocked artifact | Always emit an explicit repair boundary artifact, even when repair cannot run |
| FR-11 | Soundness clean but completeness and retention promotion blocked | Add unsolvable/hard-family curriculum and retention guard, still controller-only |
| EBT/ARM | Projection-only sidecar with tests | Measure sidecar score correlation before any integration claim |
| Sampling/hardware | CPU cLUT diagnostic only; GateMate/SSQA evidence absent | Integrate cLUT as a bounded backend path and ingest hardware evidence only |

## Three Biggest Gaps To PRD Vision

1. **Certified verifier and repair gap.** The PRD requires verifiable reasoning
   under solver/test authority. Carnot still has no positive verifier gain, no
   promoted formal-feedback loop, and no repair micro-panel artifact. `.290`
   must turn MaxSAT routing into solver-certified coherence feedback, repair
   the model-spec metadata gap, and produce an explicit repair boundary artifact
   whether repair runs or blocks.

2. **Continuous self-learning promotion gap.** FR-11 remains controller-only.
   The current controller can avoid soundness mistakes under stress, but it
   loses completeness and prior retention. `.290` must test an unsolvable or
   hard-family curriculum guard with rollback and retention gates before any
   stronger learning claim.

3. **Architecture-to-evidence bridge gap.** The long-term architecture points
   toward EBT/ARM value scoring, KAN/KAEM verification, and sampler hardware,
   but local evidence is still projection-only or diagnostic-only. `.290`
   should measure sidecar score correlation and cLUT backend integration while
   preserving no-live-integration and no-hardware-speedup claims.

## New Research Integrated

The post-`.289` planning sweep was appended to `research-references.md` before
this milestone was designed. Findings shaping `.290`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| Proof-carrying coherent reasoning can compile answer selection plus logical constraints to weighted MaxSAT with certificates, coherence gaps, and minimal repair distance | OpenReview `liNC8KHvUy` | Build formal-feedback v3 around solver-certified coherence and MCS/MUS-style localized contradictions |
| LOVER uses unsupervised logic regularization over multiple reasoning paths | arXiv:2605.05893 | Pilot a logic-regularized verifier on exact fixtures while keeping exact labels authoritative |
| NuRL identifies zero-pass hard samples as a blind spot for online RL and uses abstract self-generated hints | arXiv:2509.25666 / OpenReview `hfNnQHkTtv` | Adapt the idea only to FR-11 controller memory and curriculum, with retention and soundness gates |
| KAN property verification can use piecewise affine abstractions and MILP bounds | arXiv:2602.06737 | Add bounded KAN verification context for follow-on work; do not make it a headline `.290` blocker |
| EBFT uses sequence-level feature matching framed as energy-based fine-tuning | arXiv:2603.12248 | Inform sidecar score-correlation diagnostics without claiming weight updates |
| Lyapunov-style energy matching offers stopping/certification criteria for samplers | arXiv:2605.05530 | Keep as sampler design context for cLUT/sidecar follow-on |
| Extropic THRML/XTR-0/Z1 and Logical Intelligence Kona remain public architecture context | Extropic / Logical Intelligence pages | Keep hardware and commercial EBM claims evidence-bounded in local artifacts |

## Architecture Direction

`.290` keeps the authority stack conservative: exact fixtures, Z3, execution
tests, and solver certificates decide correctness; local SOTA GGUF models are
candidate generators and trace sources; sidecar energy layers provide bounded
diagnostics until live integration is proven.

```text
                +-----------------------------------+
                | research-roadmap-next.yaml        |
                | exp3109 ... exp3121               |
                +-----------------+-----------------+
                                  |
                                  v
+------------------+    +--------+---------+    +-----------------------+
| .289 authorities | -> | model-spec/cache | -> | certified coherence   |
| matrix v23       |    | corrigendum      |    | feedback v3           |
| capstone v289    |    | exp3110          |    | exp3111               |
+------------------+    +--------+---------+    +-----------+-----------+
                                  |                          |
                                  v                          v
                         +--------+---------+      +---------+----------+
                         | logic-regularized| ---> | diagnostic verifier|
                         | verifier pilot   |      | calibration v5     |
                         | exp3112          |      | exp3113            |
                         +--------+---------+      +---------+----------+
                                  |                          |
                                  v                          v
                         +--------+---------+      +---------+----------+
                         | fragment/code    | ---> | explicit repair    |
                         | verifier pilot   |      | boundary artifact  |
                         | exp3114          |      | exp3115            |
                         +------------------+      +--------------------+

+------------------+    +------------------+    +-----------------------+
| FR-11 .289       | -> | unsolvable        |    | sidecar / cLUT /     |
| stress artifact  |    | curriculum guard  |    | hardware evidence    |
| exp3103          |    | exp3116           |    | exp3117-exp3119      |
+------------------+    +------------------+    +-----------+-----------+
                                                              |
                                                              v
                                                  +-----------+-----------+
                                                  | matrix v24 + capstone |
                                                  | exp3120 + exp3121     |
                                                  +-----------------------+
```

## Required SOTA Model Policy

Every `.290` experiment that invokes an LLM must include `MODEL_SPECS` and
must attempt at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE)
- `unsloth/gemma-4-31B-it-GGUF` (flagship dense)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE)

Legacy small models such as `Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear
only as CPU smoke tests. They cannot headline verifier, formal-feedback,
repair, or self-learning results. Tasks that need live models must record
cache status, exact model IDs, selected quantization/file if known, prompt
hashes, live-call counts, and whether headline evidence was skipped because no
mandated model was usable.

## Milestone Phases

### Phase A - Archive, Model-Spec Corrigendum, and Certified Feedback

**Goal:** turn `.289`'s clearer failure surface into an unblocked formal
feedback path without depending on a cached two-model SOTA pair.

- `exp3109` archives `.289`, stages `.290`, and carries forward matrix v23
  blocker authority.
- `exp3110` writes the model-spec/cache manifest corrigendum so headline local
  SOTA artifacts expose `mandatory_headline_model_ids` and cache status
  consistently.
- `exp3111` builds certified coherence feedback v3 using Z3/MaxSAT-style
  certificates, MCS/MUS localization, and solver-only baselines.

### Phase B - Verifier Recovery and Repair Boundary

**Goal:** test logic-regularized verification and diagnostic calibration on the
exact protocol, then ensure repair never disappears as a missing artifact.

- `exp3112` pilots a LOVER-inspired logic-regularized verifier over exact
  fixtures and mandated local SOTA traces when available.
- `exp3113` runs verifier calibration v5 diagnostically: it may report negative
  or zero lift, but must produce a usable gate decision.
- `exp3114` adds a fragment-level code/constraint verification pilot to expose
  localized repair targets before full candidate repair.
- `exp3115` writes the explicit repair micro-panel v4 artifact. If calibration
  does not unblock repair, the artifact must say so mechanically.

### Phase C - Continuous Self-Learning and Architecture Boundaries

**Goal:** improve FR-11's controller-only boundary and move sidecar/sampler
work from projection to measured, bounded integration.

- `exp3116` is the required continuous self-learning experiment. It tests a
  solver-derived unsolvable/hard-family curriculum guard with rollback,
  soundness, completeness, and retention gates.
- `exp3117` measures EBT/ARM sidecar score correlation against exact labels
  while preserving no-live-model-integration and no-weight-update claims.
- `exp3118` integrates the cLUT sampler as an optional CPU backend path with
  distribution checks and no hardware claim.
- `exp3119` ingests GateMate/SSQA operator evidence if present and otherwise
  keeps rerun/readback blocked.

### Phase D - Matrix and Capstone

**Goal:** close `.290` from artifacts, not intent.

- `exp3120` builds cross-corpus matrix v24 with explicit rows for model-spec
  metadata, certified feedback, repair artifact presence, FR-11 promotion, and
  architecture boundaries.
- `exp3121` writes the `.290` capstone and recommends the next milestone from
  matrix v24.

## Dependency Graph

```text
exp3109 archive
  -> exp3110 model-spec/cache corrigendum
        -> exp3111 certified coherence feedback v3
              -> exp3112 logic-regularized verifier pilot
              -> exp3113 diagnostic verifier calibration v5
                    -> exp3115 explicit repair boundary artifact
        -> exp3116 FR-11 unsolvable curriculum guard

exp3114 fragment-level verification -> exp3115 explicit repair boundary

exp3109 -> exp3117 EBT/ARM sidecar score correlation
exp3109 -> exp3118 cLUT backend integration boundary
exp3109 -> exp3119 GateMate/SSQA evidence ingestion

exp3109..exp3119 -> exp3120 matrix v24 -> exp3121 capstone
```

Structured conductor gates:

- `exp3111` requires `exp3110.sota_model_manifest_ready == true`.
- `exp3112` requires `exp3111.certified_coherence_feedback_v3_ready == true`.
- `exp3113` requires `exp3111.certified_coherence_feedback_v3_ready == true`.
- `exp3121` requires `exp3120.matrix_v24_ready == true`.

Repair is deliberately not conductor-gated on a positive verifier delta:
`exp3115` must always run and write an explicit artifact. It may internally
set `repair_run_executed=false` and `repair_unblocked=false`.

## Hardware Requirements

- **Dual RTX 3090 CUDA:** required only if `exp3112`, `exp3113`, `exp3115`, or
  `exp3116` performs live local SOTA inference. Each task must use the cached
  SOTA pattern from `scripts/experiment_template.py`, record cache status, and
  skip headline LLM claims if no mandated model is usable.
- **CPU-only:** sufficient for `exp3109`, `exp3110`, `exp3111` solver-only
  certificate work, `exp3114`, `exp3117`, `exp3118`, `exp3120`, and `exp3121`.
- **GateMate / SSQA:** no board flash, no synthesis, and no hardware command is
  required by the plan. `exp3119` may only ingest operator-provided evidence
  files. If evidence is missing, it must write a blocked artifact.
- **KV260 / PolarFire / AMD XDNA / Extropic TSU:** out of scope for `.290`
  execution. They may appear only as architecture context already tracked in
  the hardware wishlist.

## Experiment List

| ID | Title | Deliverable |
| --- | --- | --- |
| exp3109 | Archive .289 and activate .290 | `results/experiment_3109_archive_v289_activate_v290.json` |
| exp3110 | Local SOTA model-spec/cache manifest corrigendum | `results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json` |
| exp3111 | Certified coherence Z3/MCS feedback v3 | `results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json` |
| exp3112 | Logic-regularized verifier reasoning pilot | `results/experiment_3112_logic_regularized_verifier_pilot_v1.json` |
| exp3113 | Diagnostic local SOTA verifier calibration v5 | `results/experiment_3113_diagnostic_local_sota_verifier_calibration_v5.json` |
| exp3114 | Fragment-level code/constraint verification pilot | `results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json` |
| exp3115 | Explicit repair gate and micro-panel artifact v4 | `results/experiment_3115_explicit_repair_gate_micro_panel_v4.json` |
| exp3116 | FR-11 unsolvable curriculum and retention guard | `results/experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.json` |
| exp3117 | EBT/ARM sidecar score-correlation boundary v3 | `results/experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.json` |
| exp3118 | cLUT sampler backend integration boundary v2 | `results/experiment_3118_clut_sampler_backend_integration_boundary_v2.json` |
| exp3119 | GateMate/SSQA operator evidence ingestion v4 | `results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json` |
| exp3120 | Cross-corpus matrix v24 | `results/experiment_3120_cross_corpus_matrix_v24.json` |
| exp3121 | Capstone v290 | `results/experiment_3121_capstone_v290.json` |

## Acceptance Criteria

`.290` succeeds if the capstone can answer these questions from artifacts:

1. Does every local SOTA headline artifact expose the mandated model IDs and
   cache status without a matrix metadata gap?
2. Does formal-feedback v3 produce a solver-certified coherence artifact with
   vacuity guards and a solver-only baseline?
3. Does verifier calibration v5 run diagnostically and produce a gate decision
   even if lift is non-positive?
4. Does the repair micro-panel v4 artifact exist and explicitly state whether
   repair ran, blocked, or should remain retired?
5. Does FR-11 improve completeness/retention without soundness mistakes, or
   correctly remain controller-only and blocked from promotion?
6. Do sidecar, cLUT, and hardware tasks keep no-live-integration and
   no-hardware-speedup boundaries honest?
7. Does matrix v24 reduce blockers or explain why the count remains unchanged?

## Failed-Rerun Compliance

Every task whose scope overlaps a retired, failed, blocked, or bounded prior
experiment includes a `prior_failures` entry in `research-roadmap-next.yaml`
with all four required fields:

- `experiment_id`
- `verdict`
- `addressed_by`
- `retire_if_same_verdict: true`

No task reuses a retired experiment ID. No `requires` or `gated_on` chain points
to a retired upstream experiment.

## Decentralization Implications

`.290` continues to advance the decentralized-agent thesis only through
verifiable local artifacts:

- local SOTA GGUFs stay optional candidates, not authorities;
- solvers and exact fixtures remain portable correctness witnesses;
- model cache and hardware access are recorded as local availability facts;
- missing operator hardware evidence blocks hardware claims rather than
  weakening the claim boundary.

## Out of Scope

- Training or fine-tuning any mandated SOTA GGUF.
- Claiming live EBT/ARM model integration.
- Claiming GateMate, SSQA, KV260, PolarFire, AMD XDNA, or Extropic TSU speedup
  without authenticated local execution evidence.
- Publishing paper-ready status unless matrix v24 and capstone v290 remove or
  explicitly downgrade the remaining blockers.

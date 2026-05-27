# Research Roadmap vNEXT - Milestone 2026.05.296

**Title:** CUDA-Backed SOTA Receipt Recovery + Adaptive Verification Granularity + FR-11 Trace Memory
**Created:** 2026-05-27
**Status:** Planned
**Supersedes:** 2026.05.295 "Receipt-Backed Live SOTA Clearance + Certificate Repair Gate + FR-11 Promotion Pack"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.295 Proved

Milestone `.295` completed all 14 planned tasks (`exp3177`-`exp3190`) and
materialized every downstream blocked state instead of leaving another missing
artifact cascade. The authoritative closeout is
`results/experiment_3190_capstone_v295.json`:

| Field | Value |
| --- | --- |
| `capstone_v295_ready` | `true` |
| `paper_ready` | `false` |
| `publication_blocker_count` | `80` |
| `blocker_delta_from_v28` | `7` |
| `missing_artifact_count` | `1` |
| `local_sota_receipt_status` | `cpu_fallback_receipt_only_non_headline_clean_rerun_blocked` |
| `verifier_status` | `gated_skip_cpu_fallback_receipt_only_flagged_adversarial_controlled_invariance_passed_exact_authority_only` |
| `repair_gate_status` | `blocked_receipt_precondition` |
| `repair_ladder_status` | `materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts` |
| `fr11_self_learning_status` | `controller_memory_promotion_allowed_cross_environment_replay_passed_no_model_weight_update` |
| `hardware_sampler_status` | `diagnostic_only_thrml_local_api_smoke_no_kv260_speedup_no_tsu_kona_execution` |
| `next_top_gap` | `full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock` |

The most important `.295` forward progress was narrow but real:

- `exp3179` loaded the local `unsloth/gemma-4-26B-A4B-it-GGUF` through
  `llama_cpp` and produced two proof receipts with transcript hashes, token
  counts, model path, model-file hash, and real wall-clock time. It remained
  `cpu_fallback_receipt_only`, so `clean_rerun_allowed=false` and
  `headline_claim_allowed=false`.
- `exp3180` passed controlled invariance over 72 exact-authority rows and the
  two receipt-backed transcripts. This removes the trace/answer shortcut
  blocker but does not clear the live verifier.
- `exp3181` correctly gate-skipped clean verifier scoring because the receipt
  was CPU fallback only. It reported `live_call_count=0`, `metrics_computed=false`,
  and `flagged_adversarial=true`.
- `exp3184` and `exp3185` preserved the repair boundary:
  `repair_gate_state=blocked_receipt_precondition`, `repair_attempt_count=0`.
- `exp3186` and `exp3187` promoted the FR-11 controller-memory update with
  heldout and cross-environment drift replay, `negative_control_regression_count=0`,
  `rollback_triggered=false`, and no model-weight update claim.
- `exp3188` proved local THRML factor-graph construction over exact rows but
  explicitly denied sampler speedup, TSU/Kona execution, and hardware claims.

The new problem is also explicit: publication blockers rose from 73 to 80
because CPU fallback receipts are useful wiring evidence but not clean local
SOTA verifier evidence. `.296` must repair the CUDA/offload path or classify
the exact blocker without re-running doomed repair tasks.

## Three Biggest Gaps To PRD Vision

1. **FR-12 verifier trust needs full local SOTA receipts, not CPU fallback
   smoke.** The project now has a proof-of-execution receipt contract and a
   known local mandated GGUF, but the clean verifier cannot run until llama.cpp
   GPU offload or another compliant CUDA path is healthy. This is the top gap.

2. **The repair loop is still blocked by verifier eligibility.** Exact rows,
   controlled invariance, and counterexample certificates exist, but repair
   calls under a flagged verifier would be a false claim. `.296` must separate
   repair-gate materialization from live repair execution and use gates to skip
   expensive repair work unless the clean verifier is actually eligible.

3. **FR-11 is controller-memory promotable but not yet a reusable
   self-verification policy.** `.295` promoted a controller-memory update, but
   the PRD vision needs a loop that learns verification traces, routes future
   checks more efficiently, and protects against forgetting. The next step is a
   VeriFY-style trace-memory controller with heldout, drift, and negative-control
   replay.

## New Research Integrated

The post-`.295` sweep was appended to `research-references.md` before this
roadmap was designed. The experiments below incorporate the strongest findings:

- **VG-Search / adaptive verification granularity** (arXiv:2505.11730) motivates
  `exp3195`, a policy that decides when to verify final answers, step chunks,
  or counterexample fragments instead of using a fixed verify-every-step rule.
- **GenCP / constraint propagation** (arXiv:2505.24012) motivates `exp3196`, a
  domain-preview compiler that constrains repair candidate spaces before any
  local SOTA generation.
- **VeriFY self-verification traces** (arXiv:2602.02018) motivates `exp3200`,
  the mandatory continuous self-learning experiment.
- **ExVerus counterexample reasoning** (arXiv:2603.25810) motivates `exp3197`,
  extending certificates into inductive invariants before repair.
- **Potts MFC and PAOA** (arXiv:2602.04200, arXiv:2507.07420) motivate
  `exp3202`, a sparse multi-state factor-graph boundary with no speedup claim.
- **PipeSD dual-threshold verification triggering** (arXiv:2605.13319)
  motivates the `.296` receipt contract: one threshold proves execution, a
  stricter threshold unlocks clean verifier and repair claims.

## v296 Architecture: Receipt-First Verifier With Adaptive Repair Control

```text
             Completed .295 exact authority + receipts
                            |
                            v
          Receipt/adversarial contract v4 (exp3192)
                            |
                            v
        llama.cpp CUDA/offload health probe (exp3193)
          |                 |                 |
          | CPU/blocked     | CUDA clean      |
          v                 v                 |
   Classified blocker   Clean SOTA verifier rerun v11 (exp3194)
          |                 |                 |
          +-----------------+-----------------+
                            |
                            v
      Adaptive verification granularity policy (exp3195)
                            |
                            v
      GenCP domain preview + ExVerus invariant certificates
              (exp3196)                 (exp3197)
                    \                     /
                     \                   /
                      v                 v
                  Repair gate v5 (exp3198)
                            |
             +--------------+--------------+
             |                             |
             v                             v
  Gated-skip repair artifact     Live repair ladder v6 (exp3199)
  with explicit blockers         only if verifier gate is clean

FR-11 side lane:
  .295 controller-memory promotion -> VeriFY trace-memory controller (exp3200)
  -> KAN-CL nonforgetting sidecar audit (exp3201)

Hardware side lane:
  THRML exact-row API boundary -> sparse Potts/MFC/PAOA factor boundary (exp3202)
  -> no KV260/GateMate/TSU/Kona speedup claim without authenticated transcript

Aggregation:
  exp3203 matrix v30 -> exp3204 capstone v296
```

The core invariant is unchanged: exact solvers, canonical answers, and
regression rows are acceptance authority. Local SOTA GGUFs can generate
receipts, proposals, or repairs only when the artifact records the actual
substrate used. Diagnostic EBMs, KANs, Potts factors, and THRML graphs stay
sidecars until they are calibrated against exact labels and supported by live
execution.

## Required SOTA Model Policy

Every `.296` task that performs or may perform an LLM call must include at
least one mandated local SOTA GGUF in its `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models such as `Qwen3.5-0.8B` or `gemma-4-E4B-it` may appear only
as loud CPU smoke fallbacks and must not populate headline-result fields.
Planned LLM-touching tasks: `exp3193`, `exp3194`, and `exp3199`.

## Milestone Phases

### Phase 1 - Archive, Contract, and CUDA Receipt Probe

- `exp3191` archives `.295` and confirms `.296` planning authority.
- `exp3192` upgrades the receipt contract to v4 with dual thresholds:
  proof-of-execution sufficiency vs clean-rerun eligibility. It also repairs
  methodology/adversarial-field discipline for aggregation and gated-skip
  artifacts.
- `exp3193` probes `llama_cpp` CUDA/offload health for the mandated GGUFs. It
  either produces a full CUDA-backed receipt that can unlock the clean verifier
  or a precise blocker (`cuda_backend_absent`, `gpu_offload_unhealthy`,
  `cache_missing`, etc.).

### Phase 2 - Clean Verifier and Adaptive Scheduling

- `exp3194` runs the clean local SOTA verifier rerun v11 only when `exp3193`
  reports `clean_rerun_allowed=true`; otherwise conductor gating should skip the
  expensive call.
- `exp3195` builds a VG-Search-inspired adaptive verification granularity
  policy over exact rows, receipts, false-accept families, and repair contexts.
- `exp3196` materializes a GenCP-style repair domain preview compiler so any
  later repair call receives a bounded candidate space and explicit constraints.

### Phase 3 - Counterexample-Guided Repair Gate

- `exp3197` turns `.295` counterexample certificates into ExVerus-style
  inductive invariant records.
- `exp3198` makes the repair-gate v5 decision from receipt, clean verifier,
  granularity, domain preview, and invariant certificates.
- `exp3199` runs a small multi-turn repair ladder only if `exp3198` unblocks
  repair; otherwise structured gating should skip the LLM call.

### Phase 4 - Self-Learning, Hardware Boundary, Matrix, and Capstone

- `exp3200` is the mandatory continuous self-learning experiment: a VeriFY-style
  self-verification trace-memory controller with heldout, drift, and
  negative-control replay.
- `exp3201` audits KAN-CL-style per-knot/nonforgetting sidecar behavior against
  exact rows and the new trace-memory controller.
- `exp3202` translates exact-row/certificate graphs into sparse q-state Potts
  and PAOA/THRML-ready factor records, denying speedup claims.
- `exp3203` writes cross-corpus matrix v30.
- `exp3204` writes the `.296` capstone and next-gap recommendation.

## Dependency Graph

```text
exp3191
  -> exp3192
       -> exp3193
            -> exp3194  [gated: clean_rerun_allowed == true]
                 -> exp3198
                      -> exp3199 [gated: repair_gate_state == "unblocked"]

exp3192 -> exp3195 -> exp3198
exp3195 -> exp3196 -> exp3198
exp3183 (.295) -> exp3197 -> exp3198

exp3186/exp3187 (.295) -> exp3200 -> exp3201

exp3188 (.295) -> exp3202

exp3193, exp3194, exp3195, exp3196, exp3197,
exp3198, exp3199, exp3200, exp3201, exp3202
  -> exp3203
       -> exp3204
```

## Hardware Requirements

| Track | `.296` use | Boundary |
| --- | --- | --- |
| Dual RTX 3090 / CUDA | Required for `exp3193` to unlock headline local SOTA claims and for `exp3194`/`exp3199` if gates pass. | If CUDA/offload is absent or unhealthy, write a blocked artifact and do not unlock repair. |
| Mandated GGUF cache | `exp3193` checks all three mandated GGUF IDs and prefers a CUDA-capable path. | CPU fallback can prove wiring only; it cannot become headline clean-verifier evidence. |
| llama.cpp / `llama_cpp` | Primary local loader path because `.295` proved it can load `gemma-4-26B-A4B-it-GGUF` on CPU. | Must record `n_gpu_layers`, backend flags, stderr tail, token counts, wall-clock, transcript hashes, and whether GPU layers were actually used. |
| KV260 / GateMate / PolarFire | No board command is required. `exp3202` only prepares sparse factor records. | No latency or speedup claim without authenticated board transcript. Do not revive retired host-SD-card workflows. |
| THRML / Extropic / Kona | `exp3202` may use local THRML APIs and public architecture context. | No TSU, Z1, XTR-0, Kona, or hardware execution claim without authenticated access and transcript. |

## Success Criteria

Minimum success for `.296`:

1. All 14 tasks either write deliverable artifacts or are structurally skipped by
   `gated_on` with an explicit gate artifact.
2. `exp3193` ends in one of two honest states: full CUDA/offload-backed
   mandated-GGUF receipt with `clean_rerun_allowed=true`, or a precise blocked
   substrate classification that preserves CPU fallback as non-headline.
3. If `exp3193.clean_rerun_allowed=true`, `exp3194` reports live clean verifier
   metrics with exact authority, methodology fields, and adversarial status. If
   false, `exp3194` must be gate-skipped rather than burning an LLM call.
4. Repair remains impossible unless `exp3198.repair_gate_state == "unblocked"`.
5. `exp3200` advances FR-11 with a controller-memory trace policy, not a hidden
   model-weight update, and passes heldout plus drift replay with zero
   negative-control regressions.
6. `exp3202` produces a sparse Potts/PAOA/THRML boundary artifact with density
   estimates and no hardware-speedup claim.
7. Matrix v30 and capstone v296 preserve paper-v6 narrowing: no KV260 speedup,
   no TSU/Kona execution, no deployed sidecar verifier, no model-weight
   self-learning, and no paper-ready claim unless the matrix supports it.

## Out Of Scope

- Editing `research-roadmap.yaml` during planning.
- Modifying `scripts/research_conductor.py`.
- Pushing commits.
- Using legacy small models as headline-result models.
- Claiming repair success from schema validity, trace polish, or CPU fallback
  receipts.
- Claiming KV260/GateMate/PolarFire/THRML/TSU/Kona speedup without an
  authenticated run transcript.

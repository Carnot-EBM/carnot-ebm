# Research Roadmap vNEXT - Milestone 2026.05.293

**Title:** Verifier Evidence Corrigendum + Repair Ladder Execution + FR-11 Ledger Closure
**Created:** 2026-05-26
**Status:** Planned
**Supersedes:** 2026.05.292 "False-Accept Verifier Recovery + Repair Gate + FR-11 Verified Memory"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.292 Proved

Milestone `.292` completed the scheduled queue, but its headline path remains
blocked. The authoritative closeout is
`results/experiment_3148_capstone_v292.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=55`
- `blocker_delta_from_v25=9`
- `next_top_gap=false_accept_recovery_corrigendum_repair_gate`
- `false_accept_recovery_status=blocked_by_adversarial_corrigendum_false_accept_0.0_known_rows_blocked`
- `live_verifier_status=flagged`
- `repair_gate_status=blocked_repair_gate_state_blocked_other_blockers_6_disqualifiers_6`
- `repair_ladder_status=gated_skipped_missing_artifact`
- `fr11_self_learning_status=bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667`
- `ebt_arm_status=projection_only_no_live_integration_blockers_6`
- `kan_status=bounded_monitor_records_2_no_deployed_verifier`
- `sampler_hardware_status=blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8`

The good news is narrow but important. `.292` found the two known `.291`
false-accept rows, then blocked both through exact accept/abstain policy and
canonical grounding:

- `exp3136` identified the source false-accept rows
  `resyn-3084-arith-003` and `resyn-3084-smt-000` with
  `source_false_accept_rate=0.5`.
- `exp3137` replayed an exact-safe accept/abstain contract and blocked the
  known false accepts with `replay_false_accept_rate=0.0`.
- `exp3138` replayed canonical answer and grounding checks and again blocked
  both known false-accept rows.
- `exp3140` produced the conservative repair gate decision
  `repair_gate_state=blocked_other`; this is the correct outcome while upstream
  verifier evidence is adversarially flagged.

The bad news is also actionable. `exp3139` reported
`false_accept_rate=0.0` and `verifier_gain_delta=0.5`, but the artifact is
itself flagged: it claims live inference while its preflight says no model was
loaded, has too-short duration for six local SOTA calls, and is missing
methodology fields. That makes `.292` a verifier recovery *corrigendum*
milestone, not a repair-unlock milestone.

| Area | `.292` result | `.293` consequence |
| --- | --- | --- |
| False-accept rows | Known rows identified and blocked by exact/canonical policy | Preserve them as non-negotiable regression rows |
| Live verifier | Source metric improved to 0.0 but artifact is flagged adversarial | Run authenticity/methodology corrigendum before trusting any live rerun |
| Repair gate | Correctly stayed `blocked_other` | Unlock only after clean verifier evidence, then execute repair ladder |
| Repair ladder | Gated skipped and artifact missing | New repair task must include prior-failure block and structured gate |
| FR-11 | Controller/environment memory only, ledger 0.833333 and 0.666667 | Close ledger consistency before promotion; no model-weight claim |
| EBT/ARM | Projection-only sidecar | Use energy diagnostics only after exact-label calibration |
| KAN | Bounded monitor records, no deployed verifier | Expand proof-carrying monitor records without deployment claim |
| Hardware | No authenticated speedup evidence | Continue evidence ingestion only; no board or TSU speedup claim |

## Three Biggest Gaps To PRD Vision

1. **Verifier evidence is not yet trustworthy enough for FR-12.** The PRD
   requires deterministic verification of violated constraints. `.292` blocked
   the known false accepts, but the live rerun artifact is flagged for evidence
   quality. The next milestone must separate model-call authenticity,
   methodology completeness, exact-label replay, and final metric claims.

2. **Repair remains blocked by verifier trust.** The repair ladder cannot
   become a headline capability until the gate is clean. `.293` adds a
   preflighted live rerun, a conservative unlock decision, and a gated repair
   ladder whose candidates are accepted only by exact tests, monitor replay, and
   counterexample evidence.

3. **Continuous self-learning has memory but not promotion-grade consistency.**
   FR-11 is still controller/environment memory only. `.293` must close ledger
   consistency to 1.0 or explicitly block promotion, then add a bounded
   attractor/residual memory audit without claiming model-weight learning.

## New Research Integrated

The post-`.292` sweep was appended to `research-references.md` before this
roadmap was designed. Findings shaping `.293`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| Sanity checks for hallucination detectors | arXiv:2605.08346 | `exp3150` audits verifier artifacts for tautology, duration, methodology, and benchmark-artifact failure modes |
| TraceFix counterexample repair | arXiv:2605.07935 | `exp3155` pilots exact counterexample-guided repair using TLA+/Z3-style evidence |
| LogicVault persistent Z3 belief state | OpenReview ICLR 2026 Logical Reasoning | `exp3156` and `exp3157` use persistent symbolic memory only when ledger replay is perfect |
| Energy-Based Constraint Networks | arXiv:2605.00960 | `exp3158` keeps energy violation localization as a calibrated diagnostic sidecar |
| Token-level and first-token hallucination signals | arXiv:2605.12384 and HF 2605.05166 | `exp3150` and `exp3152` may record cheap suspicion signals, never acceptance authority |
| Equilibrium/Attractor reasoning | HF 2605.21488 and arXiv:2605.12466 | `exp3157` audits bounded residual/attractor memory for FR-11 controller use |
| Extropic THRML/TSU and Kona/Aleph updates | vendor public pages | `exp3160` tracks evidence boundaries only, without unauthenticated speedup claims |

## Architecture Direction

`.293` keeps exact authority at the center. Local SOTA GGUF models may propose
verifier verdicts or repair candidates, but they cannot authorize acceptance.
The key architectural change is an evidence-quality layer before repair.

```text
                 +---------------------------------------------+
                 | .292 capstone                               |
                 | paper_ready=false, blockers=55              |
                 | top gap: verifier corrigendum + repair gate |
                 +----------------------+----------------------+
                                        |
                                        v
           +----------------------------+----------------------------+
           | exp3149 archive + exp3150 adversarial evidence audit    |
           +----------------------------+----------------------------+
                                        |
                                        v
                 +----------------------+----------------------+
                 | exp3151 live inference authenticity preflight |
                 +----------------------+----------------------+
                                        |
                                        v
                 +----------------------+----------------------+
                 | exp3152 clean live SOTA verifier rerun v8    |
                 +----------------------+----------------------+
                                        |
                                        v
                 +----------------------+----------------------+
                 | exp3153 repair gate unlock decision v2       |
                 +----------------------+----------------------+
                                        |
                    unblocked only      |
                                        v
        +-------------------------------+-------------------------------+
        | exp3154 repair ladder v3 + exp3155 counterexample repair pilot |
        +-------------------------------+-------------------------------+

        +------------------------+     +-----------------------------+
        | exp3156 FR-11 ledger   | --> | exp3157 attractor/memory    |
        | consistency closure    |     | residual audit              |
        +------------------------+     +-----------------------------+

        +------------------------+     +-----------------------------+
        | exp3158 EBCN energy    |     | exp3159 KAN proof-carrying  |
        | sidecar calibration    |     | monitor expansion           |
        +-----------+------------+     +---------------+-------------+
                    \                                  /
                     \                                /
                      v                              v
                 +----+------------------------------+----+
                 | exp3160 hardware boundary evidence      |
                 | exp3161 matrix v27 + exp3162 capstone   |
                 +------------------------------------------+
```

## Required SOTA Model Policy

Every `.293` experiment that invokes a local LLM must include `MODEL_SPECS`
and must attempt at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE)
- `unsloth/gemma-4-31B-it-GGUF` (flagship dense)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE)

Legacy small models such as `Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear
only as CPU smoke tests. They cannot headline verifier, repair, or
self-learning results. If a mandated model is not locally usable, the task must
write a complete blocked or diagnostic artifact with `live_call_count=0`,
`headline_claim_allowed=false`, and explicit precondition evidence.

## Milestone Phases

### Phase A - Archive And Verifier Evidence Corrigendum

**Goal:** carry `.292` forward exactly and fix the evidence-quality layer before
another live rerun is allowed to unlock repair.

- `exp3149` archives `.292`, records the 55 blockers, and stages `.293`
  without editing `research-roadmap.yaml`.
- `exp3150` performs an adversarial flag and sanity-check corrigendum across
  `exp3136`, `exp3139`, `exp3140`, `exp3142`, `exp3147`, and `exp3148`.
- `exp3151` writes a local SOTA live-inference authenticity preflight contract:
  model-load evidence, duration floor, transcript hashes, seed/checksum fields,
  and an honest blocked path.

### Phase B - Clean Verifier Rerun And Repair Gate

**Goal:** produce a clean live verifier artifact or keep the repair gate closed
without spending repair calls.

- `exp3152` reruns the local SOTA verifier panel v8 only after the evidence
  audit and authenticity preflight pass. Known `.291` false-accept rows remain
  mandatory regression rows.
- `exp3153` writes the repair-gate unlock decision v2. It may output
  `unblocked` only if the verifier rerun is not flagged and passes exact
  false-accept gates.
- `exp3154` runs the multi-turn repair ladder v3 only when `exp3153` says
  `repair_gate_state=unblocked`.
- `exp3155` adds a deterministic counterexample-guided repair pilot inspired by
  TraceFix: candidates are repaired from exact/TLA+/Z3 counterexamples, not from
  model confidence.

### Phase C - FR-11 Continuous Self-Learning Closure

**Goal:** advance autonomous self-learning without claiming weight updates or
unverified memory promotion.

- `exp3156` closes the FR-11 ledger consistency gap by replaying VeRA/EvoEnv
  and experience-memory artifacts against fresh and prior variants. Promotion is
  allowed only at `ledger_consistency_rate=1.0`.
- `exp3157` audits bounded attractor/residual memory for the controller using
  exact-label replay. It is a continuous self-learning experiment: memory can
  update routing or residual thresholds, but model weights remain unchanged.

### Phase D - Bounded Architecture And Evidence Closeout

**Goal:** keep promising architecture lanes current while preventing unsupported
headline claims.

- `exp3158` calibrates an Energy-Based Constraint Network style sidecar against
  exact false-accept and clean rows, reporting violation localization only as a
  diagnostic.
- `exp3159` expands KAN/PWA proof-carrying monitor records, still bounded and
  not a deployed verifier.
- `exp3160` ingests hardware/sampler evidence boundaries for CUDA, KV260,
  GateMate, PolarFire, Extropic, and Kona without running board commands or
  making speedup claims.
- `exp3161` writes cross-corpus matrix v27.
- `exp3162` writes the `.293` capstone and next-gap recommendation.

## Dependency Graph

```text
exp3149
  -> exp3150
  -> exp3151
  -> exp3152
  -> exp3153
  -> exp3154
       -> exp3155

exp3156
  -> exp3157

exp3150
  -> exp3158
  -> exp3159

exp3160

exp3154, exp3155, exp3157, exp3158, exp3159, exp3160
  -> exp3161
  -> exp3162
```

Structured conductor gates are used where they can avoid wasted model calls:

- `exp3152` gates on `exp3150.adversarial_corrigendum_v1_ready == true` and
  `exp3151.live_inference_authenticity_preflight_ready == true`.
- `exp3153` gates on `exp3152.clean_live_verifier_rerun_v8_ready == true`,
  `exp3152.flagged_adversarial == false`, and
  `exp3152.false_accept_gate_passed == true`.
- `exp3154` gates on `exp3153.repair_gate_state == "unblocked"`.
- `exp3155` gates on `exp3154.multi_turn_repair_ladder_v3_ready == true`.
- `exp3157` gates on `exp3156.fr11_ledger_consistency_closure_v1_ready == true`.
- `exp3162` gates on `exp3161.matrix_v27_ready == true`.

## Hardware Requirements

No new hardware claims are required for `.293`.

- **GPU/local GGUF:** `exp3151`, `exp3152`, and `exp3154` require local SOTA
  inference if they are not honestly blocked. They must record model path,
  load evidence, transcript hashes, duration, token counts, seed, and
  reproducibility checksum. Legacy small models are smoke tests only.
- **Dual RTX 3090 / CUDA:** usable only when detected and recorded by the
  task's preflight. No speedup claim is allowed without matched baseline,
  command transcript, and artifact checksum.
- **KV260, GateMate, PolarFire:** evidence ingestion only unless the operator
  has already supplied authenticated logs. Do not run board commands or convert
  wish-list status into measured speedup.
- **Extropic THRML/TSU and Kona/Aleph:** public pages remain architecture
  references. They do not support local hardware acceleration claims.

## Success Criteria

`.293` succeeds if it produces a clean, auditable answer to the repair question:

1. If live SOTA evidence is clean and false accepts remain blocked, the repair
   gate unlocks and `exp3154` produces exact-authority repair metrics.
2. If live SOTA evidence is still blocked or flagged, repair remains skipped by
   structured conductor gates and no model calls are wasted.
3. FR-11 either reaches promotion-grade ledger consistency or explicitly
   remains controller/environment memory only with replayable counterexamples.
4. Matrix v27 and the capstone state whether paper readiness improved, regressed,
   or stayed blocked, with no unsupported hardware, EBT, KAN, or live-verifier
   claims.

# Research Roadmap vNEXT - Milestone 2026.05.292

**Title:** False-Accept Verifier Recovery + Repair Gate + FR-11 Verified Memory
**Created:** 2026-05-26
**Status:** Planned
**Supersedes:** 2026.05.291 "Live SOTA Verifier Repair + FR-11 EvoEnv + Bounded Energy Monitors"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.291 Proved

Milestone `.291` completed every scheduled task, but it moved the project
farther from paper readiness because the live verifier gate exposed a real
false-accept problem. The authoritative closeout is
`results/experiment_3134_capstone_v291.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=46`
- `blocker_delta_from_v24=10`
- `next_top_gap=live_verifier_false_accept_repair_gate`
- `sota_cache_status=bounded_missing_comparative_sota_pair`
- `verifier_claim_status=blocked_false_accept_rate_0.5_no_headline_lift`
- `repair_claim_status=blocked_repair_ladder_gate_failed_by_live_verifier_gate`
- `fr11_self_learning_status=bounded_controller_environment_memory_only_no_weight_update_ledger_0.666667`
- `ebt_arm_status=projection_only_sidecar_diagnostic_no_live_integration`
- `kan_status=bounded_pwa_milp_abstraction_no_deployed_verifier_claim`
- `sampler_hardware_status=blocked_hardware_sampler_boundary_no_speedup_claim`

The most important result is negative and actionable. `exp3124` ran 6 live
calls using the only locally present mandated SOTA GGUF,
`unsloth/gemma-4-26B-A4B-it-GGUF`, and found `false_accept_rate=0.5`,
`false_reject_rate=0.0`, `verifier_gain_delta=0.0`, and
`repair_gate_state=blocked_false_accept`. The failed gate correctly prevented
`exp3127` from spending model calls on repair.

The supporting artifacts narrowed the problem. `exp3125` produced a bounded
prefix-closed verifier pilot with `bound_width=0.0020155392` over three small
fixtures. `exp3126` showed `ledger_consistency_rate=0.666667` across monitored
fixtures, with contradiction rather than satisfiable drift in observed
violations. FR-11 advanced to solver-only executable environment memory:
`exp3128` admitted 3 of 5 candidate environments with zero soundness and
completeness errors, and `exp3129` recommended controller/environment-memory
promotion only while blocking model-weight-learning claims until ledger
consistency reaches 1.0.

| Area | `.291` result | `.292` consequence |
| --- | --- | --- |
| Live verifier | 50% false accepts on bounded live panel | Diagnose row-level false-accept mechanisms before another repair attempt |
| Repair | Repair ladder gate-blocked, correctly | Unlock repair only after a strict false-accept contract passes |
| Local SOTA cache | Only Gemma-4-26B present; Qwen3.6 and Gemma-4-31B missing | Keep cache/precondition manifest, but do not make cache repair the central milestone |
| Prefix bounds | Small BEAVER-style pilot works, but not open-world LLM correctness | Reuse as an abstention/acceptance contract, not as a proof of full correctness |
| Fragment monitors | Monitors ready; ledger consistency only 0.666667 | Use ledger replay as a hard repair/FR-11 gate |
| FR-11 | Environment memory exists, controller-only | Harden with VeRA-style fresh variants and experience-memory routing |
| EBT/ARM | Energy-budget sidecar ready, projection-only | Calibrate sidecar against false-accept rows before integration |
| KAN | PWA/MILP abstraction passed, no deployed verifier claim | Try proof-carrying monitor attachment, still bounded |
| Hardware | No authenticated speedup evidence | Keep evidence ingestion; do not schedule board execution |

## Three Biggest Gaps To PRD Vision

1. **False accepts break verifiable reasoning.** The PRD's FR-12 requires
   deterministic verification of violated constraints. A live verifier that
   accepts invalid rows at 50% cannot support headline claims or repair. `.292`
   prioritizes a row-level false-accept autopsy, a stricter answer/step
   canonicalization contract, and a live rerun whose success gate is a low
   false-accept rate, not a pooled accuracy number.

2. **Repair remains blocked by verifier trust.** The repair pipeline should
   improve candidates under exact authority, but `.291` correctly gate-blocked
   repair because verifier prerequisites failed. `.292` separates repair-gate
   unlock from repair execution: first prove the verifier contract can reject
   false accepts, then run a bounded multi-turn repair ladder with exact
   tests/Z3, CRANE-style reasoning-preserving structure, and AdaDec-style
   uncertainty-triggered proposals.

3. **Continuous self-learning is still controller/environment memory only.**
   FR-11 asks for autonomous self-learning. `.291` admitted executable
   environments but did not reach perfect ledger consistency and made no weight
   update claim. `.292` hardens that loop with VeRA-style generated variants
   and an experience-driven verifier memory that escalates families with a
   history of false accepts while suppressing low-value redundant checks.

## New Research Integrated

The post-`.291` planning sweep was appended to `research-references.md` before
this milestone was designed. Findings shaping `.292`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| LLM verification can be harder than solving | OpenReview ICLR 2026 LLM Reasoning workshop | `exp3136` treats every live accept as suspect until exact labels and canonicalization agree |
| VeriCoT formalizes CoT steps into first-order logic and solver checks | OpenReview ICLR 2026 poster | `exp3138` adds premise/step grounding and answer canonicalization around false-accept rows |
| Self-verification must be learned/calibrated separately from generation | arXiv:2602.07594 | `exp3139` measures verification-specific calibration, not generation quality |
| VeRA generates verified equivalent and hardened variants from executable specs | arXiv:2602.13217 | `exp3142` hardens FR-11 EvoEnv with fresh solver-labeled variants |
| Experience-driven self-verification suppresses redundant checking | HF paper page for arXiv:2602.03485 | `exp3143` builds a memory-backed verifier routing policy |
| AdaDec reranks high-uncertainty code tokens with lookahead | arXiv:2506.08980 | `exp3141` uses uncertainty as candidate-generation support, never as authority |
| CRANE warns overly restrictive grammars can harm reasoning | arXiv:2502.09061 | `exp3141` keeps structured repair reasoning-preserving rather than final-answer-only JSON |
| GroundedPRM combines tree structure with external tool verification | OpenReview NeurIPS 2025 SEA | `exp3141` records step-level tool feedback without training a PRM |
| Citation hallucination audits show value of objectively verifiable facts | arXiv:2605.07723 | Deferred benchmark idea; `.292` stays focused on the current false-accept gate |
| Extropic and Kona/Aleph remain external architecture signals | Extropic / Logical Intelligence public pages | `exp3146` keeps hardware/Kona claims bounded to authenticated local evidence |

## Architecture Direction

`.292` makes the accept decision the center of the architecture. Local SOTA GGUF
models may propose verdicts or repair candidates, but an accept is allowed only
when exact labels, answer canonicalization, step/premise grounding, monitor
ledger replay, and prefix-bound/abstention policy agree.

```text
                +--------------------------------------+
                | .291 capstone + matrix v25           |
                | paper_ready=false, blockers=46       |
                | top gap: live verifier false accepts |
                +------------------+-------------------+
                                   |
                                   v
       +---------------------------+----------------------------+
       | exp3135 archive + exp3136 false-accept autopsy         |
       +---------------------------+----------------------------+
                                   |
             +---------------------+----------------------+
             |                                            |
             v                                            v
   +---------+----------+                       +---------+----------+
   | exp3137 accept/    |                       | exp3142 FR-11      |
   | abstain contract   |                       | VeRA/EvoEnv v2     |
   +---------+----------+                       +---------+----------+
             |                                            |
             v                                            v
   +---------+----------+                       +---------+----------+
   | exp3138 VeriCoT /  |                       | exp3143 experience |
   | answer canon pilot |                       | verifier memory    |
   +---------+----------+                       +---------+----------+
             |
             v
   +---------+----------+
   | exp3139 live SOTA  |
   | verifier rerun v7  |
   +---------+----------+
             |
             v
   +---------+----------+
   | exp3140 repair gate|
   | unlock decision    |
   +---------+----------+
             |
             v
   +---------+----------+
   | exp3141 multi-turn |
   | repair ladder v2   |
   +--------------------+

   +--------------------+     +--------------------+     +--------------------+
   | exp3144 EBT/ARM    |     | exp3145 KAN monitor|     | exp3146 hardware   |
   | false-accept calib |     | attachment boundary|     | evidence boundary  |
   +---------+----------+     +---------+----------+     +---------+----------+
             \                        |                         /
              \                       |                        /
               v                      v                       v
                 +--------------------+--------------------+
                 | exp3147 matrix v26 + exp3148 capstone   |
                 +------------------------------------------+
```

## Required SOTA Model Policy

Every `.292` experiment that invokes a local LLM must include `MODEL_SPECS` and
must attempt at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE)
- `unsloth/gemma-4-31B-it-GGUF` (flagship dense)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE)

Legacy small models such as `Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear
only as CPU smoke tests. They cannot headline verifier, repair, or
self-learning results. Because `.291` found only Gemma-4-26B locally present,
all live tasks must read the cache/precondition manifest first and write a
diagnostic or gated-skip artifact if no mandated model is usable.

## Milestone Phases

### Phase A - Archive and False-Accept Root Cause

**Goal:** preserve `.291` evidence and make the false-accept mechanism
machine-readable before retrying live verification.

- `exp3135` archives `.291`, carries forward matrix v25 and capstone blockers,
  and stages `.292` without editing `research-roadmap.yaml`.
- `exp3136` performs a row-level false-accept autopsy across `.291` live rows,
  answer extraction, exact labels, monitor events, prefix bounds, and prompt
  hashes.
- `exp3137` defines a stricter accept/abstain/reject contract that blocks
  known false-accept families and records thresholds as explicit policy.

### Phase B - Verifier Contract and Repair Gate

**Goal:** convert solver-certified feedback into a live verifier gate that is
safe enough to unlock repair.

- `exp3138` builds a VeriCoT/xVerify-style canonicalization and premise
  grounding pilot over the false-accept rows.
- `exp3139` reruns the difficulty-stratified live SOTA verifier panel using the
  new contract and reports whether `false_accept_rate <= 0.10`.
- `exp3140` writes a repair-gate unlock decision artifact from the rerun,
  monitors, and prefix bounds.
- `exp3141` runs a bounded multi-turn repair ladder v2 only if `exp3140`
  reports `repair_gate_state=unblocked`.

### Phase C - Continuous Self-Learning and Architecture Boundaries

**Goal:** advance FR-11 and architecture evidence without bypassing the new
accept contract.

- `exp3142` is the required continuous self-learning experiment. It hardens
  EvoEnv using VeRA-style executable variants and requires ledger consistency
  to reach 1.0 for promotion.
- `exp3143` adds an experience-driven verifier memory policy that suppresses
  redundant checks but escalates historically false-accept-prone families.
- `exp3144` calibrates EBT/ARM sidecar energy budgets against false-accept rows
  and keeps `live_integration=false` unless code integration actually ships.
- `exp3145` attaches KAN PWA/MILP proof-carrying monitor outputs to the bounded
  verifier ledger, still without a deployed verifier claim.
- `exp3146` refreshes hardware and sampler evidence boundaries without board
  execution or speedup claims.

### Phase D - Matrix and Capstone

**Goal:** close from artifacts, not intent.

- `exp3147` builds cross-corpus matrix v26 with explicit rows for false-accept
  recovery, repair-gate status, FR-11 verified memory, EBT/ARM calibration, KAN
  monitor attachment, and hardware boundaries.
- `exp3148` writes the `.292` capstone and recommends the next milestone from
  matrix v26.

## Dependency Graph

```text
exp3135 archive
  -> exp3136 false-accept autopsy
       -> exp3137 accept/abstain/reject contract
            -> exp3138 canonicalization + premise grounding
                 -> exp3139 live SOTA verifier rerun v7
                      -> exp3140 repair-gate unlock decision
                           -> exp3141 multi-turn repair ladder v2
       -> exp3142 FR-11 VeRA/EvoEnv hardening
            -> exp3143 experience-driven verifier memory
       -> exp3144 EBT/ARM false-accept calibration

exp3145 KAN proof-carrying monitor boundary
exp3146 hardware/sampler evidence boundary

exp3136, exp3137, exp3138, exp3139, exp3140, exp3141,
exp3142, exp3143, exp3144, exp3145, exp3146
  -> exp3147 matrix v26
       -> exp3148 capstone v292
```

Structured conductor gates are used where they save work:

- `exp3137` gates on `exp3136.false_accept_autopsy_v1_ready == true`.
- `exp3138` gates on `exp3137.acceptance_contract_v1_ready == true`.
- `exp3139` gates on `exp3138.canonical_grounding_pilot_v1_ready == true`.
- `exp3140` gates on `exp3139.live_verifier_rerun_v7_ready == true`.
- `exp3141` gates on `exp3140.repair_gate_state == "unblocked"`.
- `exp3143` gates on `exp3142.fr11_vera_evoenv_v2_ready == true`.
- `exp3144` gates on `exp3136.false_accept_autopsy_v1_ready == true`.
- `exp3148` gates on `exp3147.matrix_v26_ready == true`.

## Hardware Requirements

`.292` uses hardware conservatively:

- **Dual RTX 3090 CUDA:** required for `exp3139` and `exp3141` when local cache
  permits live mandated GGUF calls. `exp3136` and `exp3144` may read cached live
  traces but should not run new inference unless explicitly scoped.
- **CPU/Z3/tests:** required for the false-accept autopsy, answer
  canonicalization, prefix-bound contract, monitor replay, FR-11 executable
  variants, KAN PWA/MILP accounting, matrix, and capstone.
- **GateMate/SSQA/KV260/PolarFire:** no board execution is scheduled.
  `exp3146` may only ingest authenticated operator-provided evidence and must
  record `hardware_commands_run: []` unless summarizing an existing transcript.
- **THRML/TSU/Kona:** architecture context only. No TSU, Kona, or speedup claim
  is allowed without authenticated local execution evidence.

## Experiment List

| ID | Title | Phase | Deliverable |
| --- | --- | --- | --- |
| exp3135 | Archive .291 and activate .292 planning | A | `results/experiment_3135_archive_v291_activate_v292.json` |
| exp3136 | False-accept root-cause autopsy | A | `results/experiment_3136_false_accept_root_cause_autopsy_v1.json` |
| exp3137 | Exact-safe accept/abstain contract | A | `results/experiment_3137_exact_safe_accept_abstain_contract_v1.json` |
| exp3138 | Canonical answer and VeriCoT grounding pilot | B | `results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json` |
| exp3139 | Live SOTA verifier rerun v7 | B | `results/experiment_3139_live_sota_verifier_rerun_v7.json` |
| exp3140 | Repair gate unlock decision | B | `results/experiment_3140_repair_gate_unlock_decision_v1.json` |
| exp3141 | Multi-turn repair ladder v2 | B | `results/experiment_3141_multi_turn_repair_ladder_v2.json` |
| exp3142 | FR-11 VeRA/EvoEnv hardening | C | `results/experiment_3142_fr11_vera_evoenv_hardening_v2.json` |
| exp3143 | FR-11 experience-driven verifier memory | C | `results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json` |
| exp3144 | EBT/ARM false-accept calibration boundary | C | `results/experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.json` |
| exp3145 | KAN proof-carrying monitor boundary | C | `results/experiment_3145_kan_proof_carrying_monitor_boundary_v2.json` |
| exp3146 | Hardware and sampler evidence boundary v6 | C | `results/experiment_3146_hardware_sampler_evidence_boundary_v6.json` |
| exp3147 | Cross-corpus matrix v26 | D | `results/experiment_3147_cross_corpus_matrix_v26.json` |
| exp3148 | Capstone v292 | D | `results/experiment_3148_capstone_v292.json` |

## Acceptance Criteria

- All tasks either complete or honestly gate/skip with artifact-visible reasons.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain unchanged.
- Every local LLM task includes mandated SOTA GGUF `MODEL_SPECS` and records
  selected model IDs, cache/precondition status, live-call counts, and whether
  headline evidence is allowed.
- No live verifier headline claim is allowed unless false accepts are reduced
  to the explicit gate in `exp3139`.
- No repair experiment runs unless `exp3140.repair_gate_state == "unblocked"`.
- FR-11 includes at least one continuous self-learning artifact and separates
  controller/environment memory from model-weight learning.
- Energy, KAN, and hardware claims stay bounded to measured local artifacts.
- `research-references.md` contains the `.292` literature sweep before the
  roadmap design.

## Failed-Rerun Compliance

Every task whose scope intersects previous failures, bounded results, or
gate-blocked attempts has a `prior_failures` block in
`research-roadmap-next.yaml` with:

- `experiment_id`
- `verdict`
- `addressed_by`
- `retire_if_same_verdict: true`

No task reuses a retired experiment ID. No `requires` or `gated_on` chain points
to a retired upstream task. The repair tasks explicitly avoid unsupported
repair-headline wording and only collect bounded evidence after the verifier
false-accept gate clears.

## Out of Scope

- Editing `research-roadmap.yaml` or `scripts/research_conductor.py`
- Pushing changes
- Public landing page edits
- Closed-model headline evidence
- Promoting Gemma-4-26B single-model results as comparative SOTA-pair evidence
- TSU/Kona/hardware speedup claims without authenticated local evidence
- Board flashing, synthesis, or readback during hardware evidence ingestion
- Model-weight-learning claims for FR-11 unless a real tested weight update
  occurs and passes soundness, completeness, retention, and ledger gates

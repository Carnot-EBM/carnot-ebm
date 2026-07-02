# Research Roadmap vNEXT: V471 Receipt-Backed Structured Generation, Symbolic Certificates, and FR-11 Anchors

**Milestone:** `2026.07.471`
**Status:** Pre-staged next milestone
**Prepared:** 2026-07-01
**Predecessor:** `2026.07.470`
**Execution manifest:** `research-roadmap-next.yaml`

## Executive Summary

Milestone `2026.07.470` proved that Carnot can run clean local SOTA provenance and exact-checked
KAN/sampling/CSP traces, but it also showed the remaining bottleneck: learned or distributional energy
is not yet producing reliable utility beyond exact cheap baselines. V471 therefore shifts from broad
ranker claims to receipt-backed structured generation, solver-verified formulation selection, symbolic
certificate distillation, partition telemetry, and a stricter FR-11 self-learning attempt grounded in
verifier anchors.

The milestone avoids the retired FoVer selector/audit/residual-memory scopes. It also does not claim
hardware speedup or TSU execution. Hardware work is limited to authenticated local board transcripts and
hash-matched workload evidence.

## What V470 Proved

V470 completed with the capstone verdict:

`complete_capstone_v470_runtime_clean_exact_solver_progress_structured_energy_quarantined_fr11_no_promote_hardware_continuity`

Key results:

- Runtime provenance is clean enough for SOTA local-model experiments (`exp5124`).
- Structured reasoning pools can be generated with high parse coverage, but the V470 pool was
  adversarial-flagged for duration evidence and must not be treated as a clean substrate (`exp5125`).
- Distributional energy ranking did not beat the strongest cheap baseline (`distributional_energy_delta
  = 0.0`) and downstream audit was correctly skipped (`exp5126`, `exp5127`).
- KAN certificate explanations were sound and false-property controls worked on the tested family
  (`exp5128`).
- Adaptive HUBO 2D-PT improved optimum hit rate in exact-checked CPU simulation (`exp5129`).
- TACO held-out CSP traces preserved labels and reduced guarded effort slightly, but harmful guarded
  instances remained (`exp5130`).
- FR-11 case-policy self-learning correctly refused promotion at zero held-out delta (`exp5131`).
- Hardware continuity is alive for KV260 and PolarFire, but board speedup claims remain blocked until
  safe checked-in workloads and sample-quality evidence exist (`exp5132`).

V471 treats these as constraints, not as settled wins.

## Three Biggest Gaps Against The PRD

1. **Verifier-grounded utility is still too close to cheap exact baselines.**
   Carnot can produce exact labels and run local SOTA models, but the learned/ranked path did not improve
   over cheap baselines in V470. V471 must measure utility where the model contributes a different
   object: a structured formulation, guided decode trajectory, abstention trace, or symbolic certificate,
   not just a score over already easy answers.

2. **FR-11 self-learning lacks a positive, nonforgetting promotion route.**
   V470 correctly rolled back a no-improvement self-learning policy. V471 must test a different route:
   self-built verifier anchors and virtual exact-checkable tasks, inspired by OpenSkill and K2V, while
   preserving no-weight updates, held-out validation, nonforgetting, and rollback.

3. **Hardware remains continuity-only, not workload evidence.**
   The available boards are reachable, but Carnot lacks hash-matched timing transcripts for safe
   workloads that preserve sample-quality checks. V471 should first establish authenticated workload
   receipts and partition telemetry, then leave speedup claims for a later milestone.

## Literature Incorporated For V471

The following recent work was added to `research-references.md` before designing the tasks:

- OpenSkill (`arXiv:2606.06741`) for self-generated, verifier-anchored skill acquisition.
- K2V (`arXiv:2605.18261`) for verifiable data synthesis, adapted away from retired graph-grounding
  scopes.
- Symbolic-KAN (`arXiv:2603.23854`) for distilling KAN certificate residuals into explicit primitives.
- Solver-verified formulation generation and selection (`arXiv:2606.29366`) for non-FoVer structured
  reasoning with solver feedback.
- Reward-guided energy-based decoding (`arXiv:2605.28020`) for a narrow exact-validator guided-decoding
  retry with matched controls.
- VeriFY (`arXiv:2602.02018`) and SLOT (`arXiv:2505.04016`) for verification traces, abstention, and
  structured output scoring.
- Programmable p-bit and MIMO probabilistic-computing papers (`arXiv:2606.25313`,
  `arXiv:2601.09037`, `arXiv:2601.13542`) for partition telemetry and replica-exchange controls.
- Extropic TSU and Logical Intelligence Kona updates as architecture signals only.

## Architecture

```text
                         +------------------------------+
                         | V471 Source/Scope Audit      |
                         | refs, exclusion manifest,    |
                         | SOTA model discipline        |
                         +---------------+--------------+
                                         |
                                         v
+------------------+     +--------------+---------------+     +--------------------+
| Local SOTA GGUF  | --> | Receipt-Backed Structured     | --> | Exact Validators   |
| Qwen/Gemma MoE   |     | Pool v2                       |     | CSP/OR/TACO/JSON   |
| Gemma Dense      |     | prompts, model receipts,      |     | solver receipts    |
+------------------+     | command hashes, validator log |     +---------+----------+
                         +--------------+---------------+               |
                                        |                               |
                    +-------------------+-------------------+           |
                    |                   |                   |           |
                    v                   v                   v           v
       +------------+------+  +---------+---------+  +------+-----------+------+
       | Solver-Verified   |  | Energy-Guided     |  | Abstention/Trace       |
       | Formulation       |  | Decoding          |  | Verification           |
       | Selection         |  | exact constraints |  | coverage-risk          |
       +------------+------+  +---------+---------+  +----------+-------------+
                    |                   |                       |
                    +-------------------+-----------------------+
                                        |
                                        v
                          +-------------+--------------+
                          | Capstone Utility Decision  |
                          | promote, quarantine, retire|
                          +-------------+--------------+
                                        ^
                                        |
+------------------+     +--------------+---------------+     +--------------------+
| KAN Certificates | --> | Symbolic-KAN Distillation    | --> | Cycle Checker /    |
| V470 clean base  |     | primitive rules and margins  |     | False-Property     |
+------------------+     +------------------------------+     +--------------------+

+------------------+     +------------------------------+     +--------------------+
| HUBO 2D-PT CPU   | --> | Partition/Residual Telemetry  | --> | Authenticated      |
| V470 clean base  |     | exact checked, no speed claim |     | Board Workloads    |
+------------------+     +------------------------------+     +--------------------+

+------------------+     +------------------------------+     +--------------------+
| TACO/CSP Traces  | --> | Harm Root-Cause and Scale     | --> | OpenSkill/K2V      |
| V470 clean base  |     | exact labels, effort gates    |     | FR-11 Anchors      |
+------------------+     +------------------------------+     +--------------------+
```

## SOTA Model Policy

Every V471 experiment that invokes an LLM must include these local GGUF models in `MODEL_SPECS` and
must use the repository's cached SOTA pattern rather than headline legacy small models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as CPU smoke tests. They cannot provide headline results.

## Phase Plan

### Phase 0: Ledger And Source Hygiene

**Goal:** Make the transition from V470 to V471 auditable before running new science.

- `exp5134-archive-470-activate-471`: capture V470 outcomes, identify the missing `.470` entry in
  `research-complete.yaml` if still present, and verify that the active roadmap/conductor were not
  modified.
- `exp5135-v471-source-scope-audit`: verify that V471 references exist, map papers to tasks, check SOTA
  model discipline, and confirm no planned task matches retired FoVer or other exclusion-manifest
  scopes.

### Phase A: Receipt-Backed Structured Generation

**Goal:** Repair the substrate before testing energy or decoding claims.

- `exp5136-receipt-structured-pool-v2-v471`: regenerate a structured reasoning pool with model, prompt,
  command, duration, and validator receipts. This directly addresses the V470 duration/provenance flag.
- `exp5137-solver-verified-formulation-selector-v471`: test solver-verified formulation generation and
  selection on exact-checkable OR/CSP problems, a different route from answer-ranker reruns.
- `exp5138-ets-ebd-guided-decoding-v471`: run energy-guided decoding only after the pool is clean, with
  matched token/NFE controls and rerank-only baselines.
- `exp5139-abstention-and-verification-trace-v471`: measure structured verification traces and
  abstention as coverage-risk, harmful-answer reduction, and exact-validator agreement.

### Phase B: Symbolic Certificates And Sampler Telemetry

**Goal:** Extend the V470 clean wins without overstating them.

- `exp5140-symbolic-kan-certificate-distillation-v471`: distill KAN certificate residuals into symbolic
  primitives with cycle checks and false-property controls.
- `exp5141-hubo-partition-residual-exponent-v471`: measure partitioned HUBO/2D-PT telemetry with exact
  enumeration, boundary refresh ratios, and residual-energy exponents.
- `exp5142-taco-harm-rootcause-scale-v471`: scale TACO/CSP traces and diagnose the harmful guarded
  instances that remained in V470.

### Phase C: FR-11 Anchors, Hardware Continuity, And Capstone

**Goal:** Test a new no-weight self-learning route and convert board reachability into evidence.

- `exp5143-openskill-k2v-self-learning-v471`: use verifier anchors and virtual exact-checkable tasks for
  no-weight continuous self-learning. Promotion requires positive held-out utility and nonforgetting.
- `exp5144-authenticated-board-workload-v471`: produce hash-matched local board transcripts for KV260,
  GateMate, and PolarFire where available. It must not touch host `/dev/mmcblk*` for KV260 and must not
  claim TSU execution.
- `exp5145-capstone-v471`: aggregate all V471 artifacts, update recommendations, and preserve the
  distinction between clean utility, no-promote self-learning, continuity evidence, and blocked claims.

## Dependency Graph

```text
exp5134 archive/activate
  -> exp5135 source-scope audit
  -> exp5136 receipt-backed structured pool
       -> exp5137 solver-verified formulation selector
       -> exp5138 energy-guided decoding
       -> exp5139 abstention/verification trace

exp5128 clean KAN certificates
  -> exp5140 symbolic-KAN certificate distillation

exp5129 clean HUBO 2D-PT
  -> exp5141 partition/residual telemetry

exp5130 TACO held-out traces
  -> exp5142 TACO harm root-cause scale
       -> exp5143 OpenSkill/K2V FR-11 anchors

exp5132 hardware continuity blockers
  -> exp5144 authenticated board workload

exp5134..exp5144
  -> exp5145 capstone
```

Structured conductor gates:

- `exp5136` requires `exp5134.v470_runtime_clean == true`.
- `exp5137`, `exp5138`, and `exp5139` require `exp5136.structured_pool_v2_clean == true`.
- `exp5143` requires `exp5142.trace_suite_v2_ready == true`.
- `exp5145` requires all upstream state artifacts to exist; it should summarize blocked/skipped tasks
  rather than silently ignoring them.

## Hardware Requirements

V471 hardware work must follow the repository hardware discipline:

- **KV260:** use SSH to `kria`; do not inspect or mount host `/dev/mmcblk*`; record command transcript,
  workload hash, timing, and sample-quality evidence if a safe UIO workload exists.
- **GateMate A1:** use DirtyJTAG/openFPGALoader detection and existing working toolchain paths; do not
  depend on nonexistent `nextpnr-gatemate` commands.
- **PolarFire SoC:** use SSH reachability and hash-matched dispatch transcripts only.
- **RTX 3090 CUDA:** used for local SOTA GGUF inference when available; tasks must record the model path,
  quantization, command, wall-clock duration, and cache/provenance hashes.
- **Extropic TSU:** architecture reference only; no execution claim.

If a board is unavailable, the task must record the blocker and keep `no_speedup_claim: true`.

## Prior-Failure And Exclusion Discipline

V471 intentionally avoids retired FoVer in-domain pool, selector, audit, and residual-memory scopes.
Where V471 touches a nearby prior failure, the YAML includes `prior_failures` with the full required
four-field block and `retire_if_same_verdict: true`.

Expected prior-failure links:

- `exp5136` addresses the V470 structured-pool duration/provenance flag.
- `exp5137` addresses the V470 distributional-ranker zero-delta failure by changing the deliverable to
  solver-verified formulation selection.
- `exp5138` addresses the V466 guided-decoding underpowered result with clean SOTA provenance and fixed
  budget controls.
- `exp5143` addresses the V470 FR-11 no-promote result by using verifier anchors and virtual tasks.
- `exp5144` addresses the V470 authenticated-board blocker by requiring hash-matched safe workloads.
- `exp5145` is the routine capstone continuation and must not re-run science.

## Acceptance Criteria

V471 is successful if it produces at least one of these clean advances without violating retired-scope,
runtime, or provenance rules:

- A receipt-backed structured pool that passes adversarial verification and can support downstream work.
- A solver-verified formulation selector with positive delta over static/cheap baselines.
- A guided-decoding or abstention trace result with fixed budget controls and exact-validator authority.
- Symbolic-KAN certificate distillation that preserves soundness and detects false properties.
- Partition telemetry that maps exact-checked sampler behavior to board-ready workload requirements.
- A no-weight FR-11 self-learning run with either safe promotion or an honest rollback with a clearer
  blocker than V470.
- Authenticated local board workload transcripts that preserve no-speedup discipline.

## Non-Goals

- No FoVer in-domain selector/audit/residual-memory reruns.
- No ARC solve tasks.
- No hardware speedup headline without sample-quality evidence.
- No Extropic TSU execution claim.
- No modification of `research-roadmap.yaml`.
- No modification of `scripts/research_conductor.py`.

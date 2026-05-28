# Research Roadmap vNEXT - Milestone 2026.05.304

**Title:** Garak Availability + Abstention-Calibrated Verifier + Repair Gate Reopen

**Created:** 2026-05-28
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.303
**Execution queue:** `exp3281` through `exp3293`

## What Milestone 2026.05.303 Proved

Milestone `.303` completed the corpus-scale prompt-injection and FR-11
controller-memory work, but it did not reach paper readiness:

- `exp3267` archived `.302` and opened the `.303` corpus queue.
- `exp3268` produced a SOTA receipt methodology supplement.
- `exp3269` froze the full-corpus prompt-injection split manifest.
- `exp3270` and `exp3271` produced the remaining v4 teacher-label shards and
  a Garak/adaptive seed plan.
- `exp3272` assembled the full 15k v4 prompt-injection corpus and leakage
  ledger.
- `exp3273` evaluated the KAN sidecar at full-corpus scale. The sidecar failed
  promotion: `full_corpus_auroc=0.475326`,
  `delong_noninferiority_passed=false`, and it remained `sidecar_only=true`.
- `exp3274` could not run a real Garak red-team evaluation because
  `garak_available=false`; the gate failed with `blocked_garak_unavailable`.
- `exp3275` ran the clean verifier path but produced
  `clean_verifier_rerun_ready=false` because `abstention_rate=1.0` on the
  exact-check rows.
- `exp3276` and `exp3277` correctly stayed blocked: repair should not reopen
  until Garak and the clean verifier are both usable.
- `exp3278` passed the required continuous self-learning audit with controller
  memory only: `retention_score=0.982143`, `adaptation_score=1.0`,
  `forgetting_rate=0.017857`, and `negative_transfer_rate=0.0`.
- `exp3279` and `exp3280` confirmed the terminal state:
  `paper_ready=false`, `publication_blocker_count=105`,
  `next_top_gap=unblock_garak_redteam_eval`.

The natural next milestone is therefore not another corpus milestone. `.304`
must repair the evidence chain that blocks promotion: make Garak executable,
calibrate the clean verifier out of abstain-all behavior, bound or retire the
KAN sidecar, and reopen repair only if those gates pass.

## Three Biggest Gaps To PRD Vision

1. **The red-team gate is toolchain-blocked.** The PRD requires verifiable
   reasoning under adversarial pressure. `.303` could only report that Garak
   was unavailable. Without a real Garak/PromptInject run, prompt-injection
   mitigation evidence is not publication-grade.

2. **The clean verifier is over-conservative.** The PRD requires exact
   verification and useful abstention. `.303` clean verifier v14 abstained on
   every exact-check row, which avoids false accepts but makes the verifier
   unusable as a repair gate.

3. **Detector and repair claims are not yet bounded.** KAN failed
   full-corpus non-inferiority and repair remained blocked. Carnot needs a
   clear KAN boundary decision and a gated repair micro-panel that records
   localized verifier feedback, not a free-form regeneration claim.

## External Research Integrated

The 2026-05-28 post-`.303` sweep was added to `research-references.md` before
this roadmap was designed. The most relevant updates are:

- Garak documentation and NVIDIA's public repository show PromptInject probes
  and local model pathways, including GGUF/llama.cpp support. `.304` therefore
  treats Garak as a staged toolchain gate: install/probe manifest, smoke run,
  then full eval.
- I-CALM, conformal abstention policies, and ICLR 2026 abstention work frame
  abstention as calibrated selective prediction, motivating an abstention
  root-cause task before any verifier rerun.
- VERGE motivates localized repair feedback via formal/semantic routing and
  minimal correction subsets. `.304` uses that idea only after Garak and clean
  verification reopen the gate.
- 2025-2026 KAN security papers keep KAN plausible as an interpretable
  detector sidecar, but `.303` evidence forces a failure autopsy and boundary
  decision before any headline claim.
- XGrammar 2, TruncProof, and 2026 structured-generation work reinforce that
  schema validity is an interface guarantee, not semantic correctness.
- 2026 agent-memory work warns that raw episodes must be preserved and
  consolidation must be gated. `.304` extends FR-11 with Garak and abstention
  blocker traces while keeping the controller-memory-only boundary.
- EBT, ARM-as-EBM, Extropic TSU, and Logical Intelligence Kona remain
  strategic architecture signals. `.304` makes no hardware access, TSU speedup,
  Kona access, or foundation-EBT result claim.

## SOTA Local GGUF Policy

Any `.304` experiment that invokes an LLM for evidence must include at least
one mandated local SOTA GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The preferred implementation pattern is `cached_sota_pair(gpu_indices=(0, 1))`
from `scripts/experiment_template.py`. Legacy small models may appear only as
CPU smoke-test fallbacks. They cannot populate headline result fields and
cannot unblock clean verifier, Garak, repair, or publication-readiness gates.

## Architecture Diagram

```text
                 .303 terminal state
  full 15k v4 corpus ready, FR-11 memory audit passed,
  Garak unavailable, clean verifier abstain-all,
  KAN sidecar non-inferiority failed, repair gate closed,
  paper_ready=false, blockers=105
                              |
                              v
           exp3281 archive .303 and activate .304
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
 exp3282 Garak         exp3283 corpus/label   exp3286 clean-verifier
 install/probe         corrigendum audit      abstention root cause
        |                     |                     |
        v                     v                     v
 exp3284 Garak smoke   exp3288 KAN autopsy    exp3287 calibrated
        |              and boundary decision  clean verifier v15
        v                     |                     |
 exp3285 full Garak ----------+---------------------+
 red-team eval                                      |
        |                                            v
        +-------------------------------> exp3289 repair gate decision
                                                  |
                                                  v
                                       exp3290 SOTA repair micro-panel

 exp3285 + exp3287 + exp3288 + exp3289 blockers
        -> exp3291 FR-11 Garak/abstention memory replay

 exp3292 evidence matrix v36 -> exp3293 capstone v304
```

## Phase Plan

### Phase 1 - Handoff, Toolchain, And Evidence Hygiene

- `exp3281` closes `.303`, archives completed `.303` evidence if missing, and
  opens the `.304` queue.
- `exp3282` creates the Garak install/probe manifest. It must distinguish
  installed, importable, probe-inventory-ready, generator-adapter-ready, and
  blocked states.
- `exp3283` produces the prompt-injection corrigendum and duration/provenance
  audit. It records which `.303` labels are live-LLM, template-backed, cached,
  or non-headline due to duration/tautology flags.

### Phase 2 - Garak And Clean Verifier Unblock

- `exp3284` runs a small Garak local smoke against an available mandated SOTA
  GGUF target, gated on `exp3282.garak_runner_ready`.
- `exp3285` runs the full v4 Garak/DataFlip/adaptive red-team eval, gated on
  `exp3284.garak_smoke_ready` and `exp3283.corrigendum_ready`.
- `exp3286` diagnoses why clean verifier v14 abstained on every exact row.
- `exp3287` reruns the clean verifier with abstention calibration, gated on
  `exp3286.abstention_root_cause_identified`.

### Phase 3 - Detector Boundary And Repair Reopen

- `exp3288` performs the KAN sidecar failure autopsy and emits a boundary
  decision, gated on `exp3283.corrigendum_ready`.
- `exp3289` decides whether repair can reopen, gated on
  `exp3285.garak_redteam_eval_ready`,
  `exp3287.clean_verifier_rerun_ready`, and
  `exp3288.kan_boundary_decision_ready`.
- `exp3290` runs the SOTA repair micro-panel only if
  `exp3289.repair_gate_open` is true.

### Phase 4 - Continuous Self-Learning And Aggregation

- `exp3291` is the required continuous self-learning experiment. It replays
  Garak, abstention, KAN-boundary, and repair-gate traces through the FR-11
  controller-memory loop, preserves raw episodes, and reports retention,
  adaptation, forgetting, and negative transfer without foundation-weight
  updates.
- `exp3292` builds evidence matrix v36.
- `exp3293` produces the `.304` capstone and names the next top gap.

## Dependency Graph

```text
exp3281
  -> exp3282
      -> exp3284 [gate: garak_runner_ready == true]
          -> exp3285 [gate: garak_smoke_ready == true]

exp3281
  -> exp3283
      -> exp3285 [gate: corrigendum_ready == true]
      -> exp3288 [gate: corrigendum_ready == true]

exp3281
  -> exp3286
      -> exp3287 [gate: abstention_root_cause_identified == true]

exp3285.garak_redteam_eval_ready
  + exp3287.clean_verifier_rerun_ready
  + exp3288.kan_boundary_decision_ready
      -> exp3289
          -> exp3290 [gate: repair_gate_open == true]

exp3285 + exp3287 + exp3288 + exp3289
  -> exp3291

exp3292
  -> exp3293 [gate: matrix_v36_ready == true]
```

## Hardware Requirements

- **Dual RTX 3090 local host:** Required for live mandated SOTA GGUF tasks
  (`exp3284`, `exp3285`, `exp3287`, `exp3290`). These tasks must check
  `nvidia-smi`, selected-Python CUDA, llama.cpp/GGUF loadability, and record
  model IDs, GPU memory, duration, and generated-token counts.
- **CPU-only path:** Acceptable for handoff, corrigendum, abstention
  root-cause analysis, KAN artifact autopsy, repair-gate aggregation, FR-11
  controller-memory replay, evidence matrix, and capstone tasks.
- **Network/package access:** `exp3282` may need package metadata or a local
  isolated environment to install/probe Garak. If network/package install is
  unavailable, it must emit an explicit blocked artifact with the missing
  command and failure reason.
- **KV260/GateMate/PolarFire/THRML/Extropic/Kona:** Out of scope for `.304`.
  They remain long-term architecture signals from `research-hardware-wishlist.md`
  and `research-references.md`; this milestone makes no hardware acceleration
  or third-party proprietary access claim.

## Experiment Queue

| ID | Title | Primary Deliverable | Phase |
| --- | --- | --- | --- |
| `exp3281` | Close .303 ledger and open .304 blocker queue | `results/experiment_3281_archive_v303_activate_v304.json` | 1 |
| `exp3282` | Garak install and probe manifest v1 | `results/experiment_3282_garak_install_and_probe_manifest_v1.json` | 1 |
| `exp3283` | Prompt-injection corrigendum and duration audit v1 | `results/experiment_3283_prompt_injection_corrigendum_duration_audit_v1.json` | 1 |
| `exp3284` | Garak local smoke against mandated SOTA GGUF v1 | `results/experiment_3284_garak_local_smoke_sota_gguf_v1.json` | 2 |
| `exp3285` | Full Garak/DataFlip red-team eval v2 | `results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json` | 2 |
| `exp3286` | Clean verifier abstention root-cause audit v1 | `results/experiment_3286_clean_verifier_abstention_root_cause_v1.json` | 2 |
| `exp3287` | Abstention-calibrated clean verifier v15 | `results/experiment_3287_abstention_calibrated_clean_verifier_v15.json` | 2 |
| `exp3288` | KAN sidecar failure autopsy and boundary decision v1 | `results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json` | 3 |
| `exp3289` | Repair gate decision v9 after Garak and abstention | `results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json` | 3 |
| `exp3290` | Gated SOTA repair micro-panel v10 | `results/experiment_3290_gated_sota_repair_micro_panel_v10.json` | 3 |
| `exp3291` | FR-11 Garak/abstention memory replay v1 | `results/experiment_3291_fr11_garak_abstention_memory_replay_v1.json` | 4 |
| `exp3292` | Evidence matrix v36 | `results/experiment_3292_evidence_matrix_v36.json` | 4 |
| `exp3293` | Capstone v304 | `results/experiment_3293_capstone_v304.json` | 4 |

## Done Criteria

- Garak status is no longer ambiguous: the milestone has either a runnable
  Garak probe manifest and eval artifacts, or a precise blocked artifact with
  command-level root cause.
- Clean verifier v15 reports a non-trivial abstention profile on exact rows;
  abstain-all cannot unblock repair.
- KAN is either bounded to an explicit sidecar role with failure reasons or
  retired from the prompt-injection headline path.
- Repair micro-panel runs only through the structured gate and uses mandated
  SOTA local GGUF `MODEL_SPECS`.
- FR-11 replay includes Garak and abstention traces, reports retention,
  adaptation, forgetting, and negative transfer, and preserves the
  controller-memory-only boundary.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain
  untouched by planning.

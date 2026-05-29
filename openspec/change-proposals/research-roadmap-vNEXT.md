# Research Roadmap vNEXT - Milestone 2026.05.306

**Title:** DataFlip + Quality-Flag Cleanup For Publication-Ready Evidence

**Created:** 2026-05-29
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.305
**Execution queue:** `exp3307` through `exp3320`

## What Milestone 2026.05.305 Proved

Milestone `.305` reduced the publication blocker count but did not make the
paper ready. It proved:

- `exp3300` got the Garak gate itself to pass:
  `garak_gate_passed=true` and `attack_success_rate=0.0`.
- The same artifact did not pass the DataFlip gate:
  `dataflip_gate_passed=false`.
- Evidence matrix v37 and capstone v305 blocked promotion because the Garak
  row carried current critical quality flags: `TAUTOLOGY` and
  `DURATION_TOO_SHORT`.
- `exp3302` ran a 30-case SOTA repair panel with
  `verified_success_count=27`, `repair_success_rate=0.9`, and
  `false_accept_count=0`, but `headline_claim_allowed=false`.
- `exp3303` confirmed the repair headline remained blocked:
  `headline_claim_allowed_after_audit=false`,
  `source_provenance_clean=false`, and
  `substrate_consistency_passed=false`.
- `exp3304` kept the FR-11 controller-memory replay safe:
  `retention_score=0.982143`, `adaptation_score=1.0`,
  `forgetting_rate=0.017857`, `negative_transfer_rate=0.033333`, and
  `foundation_weight_updates_performed=false`.
- `exp3305` and `exp3306` closed `.305` with `paper_ready=false`,
  `publication_blocker_count=8`, and
  `next_top_gap=clear_garak_dataflip_and_quality_flags`.

The next milestone should not add a new benchmark family before the current
evidence is clean. `.306` must make the adversarial evidence provenance
publication-grade, rerun the repair evidence under the same runtime discipline,
and convert the observed failures into a controlled continuous self-learning
curriculum.

## Three Biggest Gaps To PRD Vision

1. **Adversarial verification still fails DataFlip and evidence hygiene.**
   The PRD requires verifiable reasoning under adversarial pressure. `.305`
   passed Garak attack-success rate but failed DataFlip and produced critical
   quality flags. The gap is no longer "can Garak run"; it is "can Carnot prove
   its prompt-injection defense works without tautological metrics or suspect
   runtime provenance."

2. **Headline repair evidence is blocked by substrate/provenance, not exact
   success count.** `.305` achieved a useful 30-case repair panel and zero false
   accepts, but the audit blocked headline use because runtime provenance and
   substrate consistency were not clean. The gap is a runtime receipt contract,
   an uncertainty-aware repair audit, and a repair rerun whose evidence can be
   promoted without special pleading.

3. **FR-11 still learns from replayed outcomes rather than a failure-targeted
   curriculum.** The controller-memory loop is safe, but it is not yet using
   DataFlip misses, quality-flag root causes, and repair audit failures as a
   directed curriculum. The PRD vision needs autonomous self-learning that
   targets verified weaknesses while preserving raw episodes and preventing
   negative transfer.

## External Research Integrated

The 2026-05-29 post-`.305` sweep was added to `research-references.md` before
this roadmap was designed. The most relevant findings are:

- **DataFlip / KAD** (`arXiv:2507.05630`) warns that output-only LLM prompt
  injection detectors can be defeated by simple data transformations. `.306`
  adds an explicit KAD/DataFlip challenge manifest and a provenance-aware guard
  before any live rerun.
- **PCFI** (`arXiv:2603.18433`), **ARGUS** (`arXiv:2605.03378`), and
  **PromptArmor** (`arXiv:2507.15219`) motivate structured prompt segments,
  authority/provenance checks, and priority-aware guard policies rather than a
  flat refusal heuristic.
- **Distributional EBMs** (`arXiv:2605.18871`) match the repair evidence gap:
  exact checks can remain the authority while a distributional energy sidecar
  represents uncertainty, abstention, and row-level provenance risk.
- **Verifier-Guided Backtracking** (OpenReview 2025) motivates a repair policy
  that backtracks when process-verifier confidence is low, while exact
  acceptance checks remain the final gate.
- **Variation in Verification** (OpenReview 2025) supports cross-family
  verification and motivates keeping Qwen/Gemma-family evidence separate
  instead of relying on one self-verification loop.
- **TTSR**, **TTCS**, and **VDS-TTT** motivate a failure-targeted curriculum for
  FR-11: train or update only from verified failures and keep controller memory
  separate from foundation weights.
- **P-bit GPU simulated annealing** and **p-dits** are promising for future
  hardware sampling work, but `.306` should only record a hardware path. It
  should make no FPGA, TSU, Kona, or proprietary-hardware speedup claim.
- **EBT**, **ARM-EBM**, **Extropic**, **Kona**, GitHub, HuggingFace Papers, and
  OpenReview checks produced watch-list items, not immediate `.306` blockers.
  Semantic Scholar citation checks for EBT (`2507.02092`) and ARM-EBM
  (`2512.15605`) were rate-limited during planning and should be retried in a
  future literature sweep.

## SOTA Local GGUF Policy

Any `.306` experiment that invokes an LLM for evidence must include at least
one mandated local SOTA GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The preferred implementation pattern is `cached_sota_pair(gpu_indices=(0, 1))`
from `scripts/experiment_template.py`, extended only when the task can safely
load the third model. Legacy small models may appear only as CPU smoke-test
fallbacks. They cannot populate headline result fields and cannot unblock
DataFlip, repair, or publication-readiness gates.

## Architecture Diagram

```text
                    .305 terminal state
 paper_ready=false; publication_blocker_count=8;
 Garak gate passed but DataFlip failed;
 current critical quality flags on Garak and repair rows;
 repair N=30 has zero false accepts but no headline permission;
 FR-11 controller memory safe but not failure-targeted
                             |
                             v
             exp3307 archive .305 and activate .306
                             |
        +--------------------+---------------------+
        |                                          |
        v                                          v
 exp3308 quality-flag                    exp3310 DataFlip/KAD
 root-cause autopsy                      challenge manifest
        |                                          |
        v                                          v
 exp3309 runtime/provenance       exp3311 PCFI/ARGUS guard pilot
 contract                         [gated on manifest]
        |                                          |
        +--------------------+---------------------+
                             v
        exp3312 live Garak/DataFlip clean rerun v4
        [gated on runtime contract + guard policy]
                             |
                             v
             +---------------+---------------+
             |                               |
             v                               v
 exp3313 repair substrate          exp3318 FR-11 failure-targeted
 autopsy                           curriculum replay
             |
             v
 exp3314 distributional EBM
 repair uncertainty audit
             |
             v
 exp3315 VGB repair backtracking policy
             |
             v
 exp3316 live SOTA repair rerun v12
 [gated on clean DataFlip + audit + policy]
             |
             v
 exp3317 repair headline audit v2
             |
             v
 exp3319 evidence matrix v38 -> exp3320 capstone v306
```

## Phase Plan

### Phase 1 - Close .305 And Define Evidence Hygiene

- `exp3307` archives `.305`, opens `.306`, and records the eight remaining
  publication blockers.
- `exp3308` performs a root-cause autopsy of `TAUTOLOGY`, `DURATION_TOO_SHORT`,
  and repair substrate/provenance failures without launching new model runs.
- `exp3309` creates an executable runtime/provenance contract for live GGUF
  artifacts: model identity, cache provenance, load timing, token counts,
  command records, duration floor, and metric-independence checks.

### Phase 2 - DataFlip And Prompt-Injection Quality Cleanup

- `exp3310` creates the DataFlip/KAD challenge manifest from `.305` failures
  and the 2025 DataFlip paper.
- `exp3311` implements a PCFI/ARGUS-style prompt provenance guard over cached
  challenge cases before any expensive live rerun.
- `exp3312` reruns the live Garak/DataFlip evaluation only when the guard and
  runtime contract are ready. This task is the milestone's primary
  adversarial-evidence promotion gate.

### Phase 3 - Repair Evidence Cleanup

- `exp3313` performs a repair-substrate autopsy of the `.305` panel and audit.
- `exp3314` adds a Distributional-EBM-inspired repair uncertainty audit sidecar
  that separates exact constraint success, learned/proxy quality, abstention,
  model identity, and provenance risk.
- `exp3315` adds a verifier-guided backtracking repair policy with exact
  acceptance authority.
- `exp3316` reruns the 30-case SOTA repair panel under the runtime contract,
  using the Distributional EBM sidecar and backtracking policy.
- `exp3317` audits the rerun for headline eligibility: provenance, duration,
  substrate consistency, confidence intervals, and false accepts.

### Phase 4 - Continuous Self-Learning And Closeout

- `exp3318` is the required continuous self-learning experiment. It turns
  DataFlip misses, quality-flag root causes, and repair audit results into a
  failure-targeted FR-11 controller-memory curriculum. It preserves raw
  episodes, reports retention/adaptation/forgetting/negative transfer, and
  performs no foundation weight updates.
- `exp3319` builds evidence matrix v38 from `.306` artifacts.
- `exp3320` closes `.306`, reports whether `paper_ready` changed, and names
  the next top gap.

## Dependency Graph

```text
exp3307
  -> exp3308
      -> exp3309 [gate: quality_flag_autopsy_ready == true]

exp3307
  -> exp3310
      -> exp3311 [gate: dataflip_manifest_ready == true]

exp3309.runtime_contract_ready == true
  + exp3311.dataflip_guard_policy_ready == true
      -> exp3312

exp3307
  -> exp3313
      -> exp3314 [gate: repair_substrate_autopsy_ready == true]
          -> exp3316 [gate: distributional_repair_audit_ready == true]
      -> exp3315
          -> exp3316 [gate: vgb_repair_policy_ready == true]

exp3312.dataflip_gate_passed == true
  + exp3312.quality_flags_cleared == true
  + exp3309.runtime_contract_ready == true
  + exp3314.distributional_repair_audit_ready == true
  + exp3315.vgb_repair_policy_ready == true
      -> exp3316
          -> exp3317 [gate: repair_rerun_v12_ready == true]

exp3312.garak_dataflip_eval_v4_ready == true
  + exp3317.repair_headline_evidence_audit_v2_ready == true
      -> exp3318

exp3312.garak_dataflip_eval_v4_ready == true
  + exp3317.repair_headline_evidence_audit_v2_ready == true
  + exp3318.fr11_failure_targeted_curriculum_ready == true
      -> exp3319
          -> exp3320 [gate: matrix_v38_ready == true]
```

## Hardware Requirements

- **Dual RTX 3090 local host:** Required for `exp3312` and `exp3316`. These
  tasks must check `nvidia-smi`, selected-Python CUDA visibility, llama.cpp or
  GGUF runtime identity, model IDs, GPU memory, generated tokens, wall-clock
  duration, and cache provenance. They must include the mandated SOTA GGUF
  model specs.
- **CPU-only path:** Acceptable for archive, autopsy, manifest, guard pilot,
  runtime contract, repair policy, FR-11 controller-memory replay, evidence
  matrix, and capstone tasks.
- **Network/package access:** Not required as a blocker. If a task needs a
  package or model that is unavailable, it must write a blocked artifact with
  exact command, environment, and stderr summaries.
- **KV260/GateMate/PolarFire:** Out of scope for `.306` execution. The FR-11
  task must name a hardware path but should not claim FPGA measurements.
- **THRML/Extropic/Kona:** Watch-list only for `.306`. No TSU, Kona, D-Wave, or
  proprietary thermodynamic hardware access claim is allowed.

## Experiment Queue

| ID | Title | Primary Deliverable | Phase |
| --- | --- | --- | --- |
| `exp3307` | Close .305 ledger and open .306 quality-cleanup queue | `results/experiment_3307_archive_v305_activate_v306.json` | 1 |
| `exp3308` | Quality-flag root-cause autopsy v1 | `results/experiment_3308_quality_flag_root_cause_autopsy_v1.json` | 1 |
| `exp3309` | Live runtime provenance contract v1 | `results/experiment_3309_live_runtime_provenance_contract_v1.json` | 1 |
| `exp3310` | DataFlip/KAD challenge manifest v1 | `results/experiment_3310_dataflip_kad_challenge_manifest_v1.json` | 2 |
| `exp3311` | PCFI/ARGUS DataFlip guard pilot v1 | `results/experiment_3311_pcfi_argus_dataflip_guard_pilot_v1.json` | 2 |
| `exp3312` | Gated DataFlip/Garak quality-clean rerun v4 | `results/experiment_3312_dataflip_garak_quality_clean_rerun_v4.json` | 2 |
| `exp3313` | Repair substrate root-cause autopsy v1 | `results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json` | 3 |
| `exp3314` | Distributional EBM repair uncertainty audit v1 | `results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json` | 3 |
| `exp3315` | VGB repair backtracking policy v1 | `results/experiment_3315_vgb_backtracking_repair_policy_v1.json` | 3 |
| `exp3316` | Gated SOTA repair rerun v12 runtime-clean | `results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json` | 3 |
| `exp3317` | Repair headline evidence audit v2 | `results/experiment_3317_repair_headline_evidence_audit_v2.json` | 3 |
| `exp3318` | FR-11 failure-targeted curriculum replay v3 | `results/experiment_3318_fr11_failure_targeted_curriculum_replay_v3.json` | 4 |
| `exp3319` | Evidence matrix v38 | `results/experiment_3319_evidence_matrix_v38.json` | 4 |
| `exp3320` | Capstone v306 | `results/experiment_3320_capstone_v306.json` | 4 |

## Done Criteria

- Garak/DataFlip has a clean v4 artifact: `garak_gate_passed=true`,
  `dataflip_gate_passed=true`, `quality_flags_cleared=true`,
  `runtime_provenance_clean=true`, and no current critical adversarial
  verification flags.
- The DataFlip defense is based on prompt provenance/priority and KAD
  transformations, not just output-only refusal scoring.
- Repair evidence is headline-eligible only if the rerun has clean runtime
  provenance, substrate consistency, confidence intervals, zero false accepts,
  and a passing audit.
- FR-11 replay uses a failure-targeted curriculum from `.306` artifacts,
  preserves raw traces, reports retention/adaptation/forgetting/negative
  transfer, and keeps the controller-memory-only boundary.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain untouched
  by planning.

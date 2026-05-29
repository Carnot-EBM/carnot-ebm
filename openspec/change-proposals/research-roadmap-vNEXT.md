# Research Roadmap vNEXT: Milestone 2026.05.309

**Title:** Runtime-Proven Energy Descent, Diversity Remediation, and Hardware Continuity

**Created:** 2026-05-29
**Status:** Proposed next milestone
**Supersedes:** 2026.05.308
**Milestone YAML:** `research-roadmap-next.yaml`
**Primary references:** `research-program.md`, `_bmad/prd.md`,
`_bmad/architecture.md`, `ops/status.md`, `ops/changelog.md`,
`ops/conductor-log.md`, `research-complete.yaml`, `research-roadmap.yaml`,
`research-references.md`, `research-hardware-wishlist.md`,
`ops/exclusion_manifest.yaml`

## What Previous Milestone Proved

Milestone `2026.05.308` completed operationally, but it did not close the
Phase-3 scientific path. The useful evidence is narrower:

- `exp3329-verifier_ensemble_diversity_audit_v2` produced a real cached audit
  with `n_cases=1000`, `effective_k=4.66`, and `lambda_min_sigma=0.0179188`.
  This is usable measurement, but the low minimum eigenvalue means the ensemble
  still needs a new independent axis before a strong diversity claim.
- `exp3331` and `exp3332` showed EBT sidecar and Interwhen-style monitor
  interfaces can emit diagnostic artifacts, but only on tiny fixtures.
- `exp3333` exercised energy-guided SOTA TT-scaling with the mandated GGUF model
  specs, but the artifact was adversarially flagged for duration/provenance
  risk. It is diagnostic, not headline evidence.
- `exp3334` gave a small FR-11 online verifier-memory nonforgetting result:
  `new_task_delta=0.05`, `old_task_delta=-0.02`, `rollback_count=2`, and
  `fr11_nonforgetting_ready=true`.

The blocked and missing pieces are more important for `.309`:

- `exp3327` reached the SOTA GGUF cache paths but blocked on tokenizer/runtime
  setup for `unsloth/Qwen3.6-35B-A3B-GGUF`; the full `exp3328` live
  energy-descent-vs-AR panel therefore did not run.
- `exp3330` diversity remediation, `exp3335` reproducer matrix, and `exp3336`
  capstone did not land as usable artifacts.
- Operator-deferred hardware priorities from `.308` remain open: KV260
  MMD-vs-CPU sequential Gibbs and GateMate n16 Ising tile build/flash smoke.

The `.309` job is therefore not to broaden claims. It is to make the SOTA
runtime, verifier diversity, FR-11 self-learning, and local hardware evidence
conductor-runnable and falsifiable.

## Three Biggest Gaps To PRD Vision

1. **Live SOTA energy-descent evidence is still substrate-blocked.** The PRD's
   verifiable reasoning vision requires real local model inference, exact
   verifier authority, and clean runtime provenance. `.308` still lacks a clean
   SOTA GGUF runtime receipt and the core energy-descent-vs-autoregressive
   comparison.

2. **Verifier diversity is measured but not remediated.** Carnot now has a
   cached covariance/effective-k audit, but `lambda_min_sigma=0.0179188` and the
   collapsed exact/symbolic pair mean the ensemble can still be one correlated
   failure mode in disguise. The next milestone must add an independent
   monitor/provenance axis and rerun the audit.

3. **FR-11 is promising but still small and not infrastructure-grade.** The
   online memory result proves the loop can update with rollback on a toy panel,
   but the PRD requires continuous self-learning under nonforgetting,
   soundness/completeness cost accounting, and reproducible artifacts. `.309`
   must scale FR-11 beyond the small controller result while keeping exact
   verifiers as ground truth.

Hardware acceleration is the fourth practical gap. It does not outrank the
science path, but the PRD/architecture explicitly require board-local evidence
for FPGA/thermodynamic sampling claims. `.309` includes the two deferred board
tasks so hardware does not drift another cycle.

## External Research Integrated

The post-`.308` sweep added a new top section to `research-references.md` before
this roadmap was written. The actionable findings are:

- **EBT / ARM-as-EBM / LoopUS / CEM:** energy-descent and looped latent
  refinement are now a strong external cluster, but Carnot should first prove
  the local SOTA GGUF runtime and exact-verifier panel before making any
  architecture claim.
- **Interwhen / ConstraintBench / BEAVER / HIVE:** the current verification
  literature points toward external deterministic authority, trajectory
  telemetry, and feasibility-first constraint checks. `.309` responds with a
  monitor/provenance verifier axis and a diversity reaudit.
- **llguidance / XGrammar / constrained speculative sampling:** structured
  generation can reduce malformed extractor outputs. Carnot should use it as a
  proposal-format layer only; semantic authority remains with exact verifiers.
- **Online CoT verifier learnability / KAN-CL / KAN verification:** FR-11 should
  account separately for soundness and completeness mistakes and use
  locality-aware or rollback-aware updates to avoid forgetting.
- **Extropic TSU / THRML and Logical Intelligence Kona/Aleph:** public hardware
  and EBM architecture signals support Carnot's long-term direction, but local
  speedup or parity claims require board-local transcripts.

## SOTA Local GGUF Policy

Every `.309` task that performs live LLM inference must declare `MODEL_SPECS`
using at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The expected implementation pattern is `cached_sota_pair(gpu_indices=(0, 1))`
from `scripts/experiment_template.py`. Legacy small models are allowed only for
explicit CPU smoke tests and cannot supply headline metrics. Live-inference
artifacts must use `inference_substrate=live_llm_inference`, record
`model_specs`, record cache paths and GPU status, and pass the duration and
provenance rules in `CLAUDE.md`.

## Architecture

```text
                         Milestone 2026.05.309

       .308 artifacts + conductor log + exclusion manifest
                              |
                              v
          Phase 0: archive missing evidence honestly
                              |
                              v
        +---------------------+----------------------+
        |                                            |
        v                                            v
+---------------------------+          +-----------------------------+
| SOTA GGUF runtime receipt |          | Diversity remediation plan  |
| Qwen3.6 / Gemma4 local    |          | exact + symbolic + monitor  |
+-------------+-------------+          +--------------+--------------+
              |                                       |
              v                                       v
+---------------------------+          +-----------------------------+
| energy-descent bootstrap  |          | monitor/provenance axis     |
| runtime-clean telemetry   |          | trajectory features         |
+-------------+-------------+          +--------------+--------------+
              |                                       |
              v                                       v
+---------------------------+          +-----------------------------+
| energy descent vs AR v3   |          | verifier diversity reaudit  |
| exact verifier authority  |          | lambda_min / effective-k    |
+-------------+-------------+          +--------------+--------------+
              |                                       |
              +-------------------+-------------------+
                                  |
                                  v
       +--------------------------+--------------------------+
       | proposal and learning layer                         |
       | constrained structured extraction + FR-11 v5         |
       | online memory, rollback, soundness/completeness      |
       +--------------------------+--------------------------+
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
   +-----------------------------+   +-----------------------------+
   | KV260 MMD vs CPU Gibbs      |   | GateMate n16 Ising tile     |
   | board-local transcript      |   | build / detect / flash      |
   +-----------------------------+   +-----------------------------+
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
             reproducer/evidence matrix v40 + capstone v309
```

## Phase Plan

### Phase 0 - Archive And Runtime Recovery

Goal: convert `.308` into an honest starting point and unblock live local SOTA
inference.

- `exp3337` archives `.308`, records missing/blocked artifacts, and activates
  `.309`.
- `exp3338` produces a SOTA GGUF tokenizer/runtime receipt for the mandated
  model set, including the Qwen3.6 blocked-tokenizer root cause if still
  present.
- `exp3339` reruns the small energy-descent bootstrap only after the runtime
  receipt is clean.

### Phase 1 - Phase-3 Energy Evidence

Goal: recover the existential Phase-3 comparison with duration-clean local SOTA
models and exact verifier authority.

- `exp3340` runs the energy-descent-vs-autoregressive SOTA panel v3, gated on
  the runtime-clean bootstrap. The experiment may produce a positive, negative,
  or blocked result; it must not use legacy tiny models for headline rows.

### Phase 2 - Verifier Diversity And Structured Extraction

Goal: turn the `.308` diversity measurement into remediation rather than another
correlated audit.

- `exp3341` writes the diversity remediation plan from `exp3329`.
- `exp3342` adds a monitor/provenance verifier axis inspired by Interwhen/HIVE
  trajectory evidence.
- `exp3343` reruns the diversity audit with the new axis and reports
  lambda_min/effective-k changes.
- `exp3344` tests constrained structured extraction with llguidance/XGrammar
  style tooling on SOTA GGUF outputs. It improves parsing only; exact verifiers
  remain semantic authority.

### Phase 3 - Continuous Self-Learning And Hardware Continuity

Goal: scale FR-11 and keep board-local acceleration evidence alive.

- `exp3345` runs FR-11 online verifier-memory nonforgetting v5 with larger
  holdouts, rollback, and soundness/completeness cost accounting.
- `exp3346` runs the deferred KV260 MMD-vs-CPU sequential Gibbs board task or
  emits a precise blocked artifact.
- `exp3347` runs the deferred GateMate n16 Ising tile build/detect/flash smoke
  or emits a precise blocked artifact.

### Phase 4 - Evidence Packaging And Decision

Goal: leave the milestone with a terminal, reproducible decision even if an
upstream blocks.

- `exp3348` creates the independent reproducer pack and evidence matrix v40.
- `exp3349` writes the capstone and next-top-gap decision.

## Task Summary

| Exp | Title | Substrate | Main deliverable |
|---|---|---|---|
| 3337 | Archive `.308`, activate `.309` | aggregation | `results/experiment_3337_archive_v308_activate_v309.json` |
| 3338 | SOTA GGUF tokenizer/runtime receipt | live LLM preflight | `results/experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1.json` |
| 3339 | Energy-descent bootstrap v2 | live LLM | `results/experiment_3339_energy_descent_bootstrap_v2_runtime_clean.json` |
| 3340 | Energy-descent vs AR SOTA panel v3 | live LLM | `results/experiment_3340_energy_descent_vs_ar_panel_v3.json` |
| 3341 | Diversity remediation plan v2 | aggregation | `results/experiment_3341_verifier_diversity_remediation_plan_v2.json` |
| 3342 | Monitor/provenance verifier axis | cached verifier | `results/experiment_3342_monitor_provenance_verifier_axis_v1.json` |
| 3343 | Verifier diversity reaudit v3 | verifier ensemble | `results/experiment_3343_verifier_diversity_reaudit_after_axis_v3.json` |
| 3344 | Constrained structured extraction smoke | live LLM | `results/experiment_3344_constrained_output_extractor_llguidance_smoke_v1.json` |
| 3345 | FR-11 online memory nonforgetting v5 | cached verifier | `results/experiment_3345_fr11_online_memory_nonforgetting_v5.json` |
| 3346 | KV260 MMD vs CPU sequential Gibbs | hardware | `results/experiment_3346_kv260_mmd_vs_cpu_sequential_gibbs_v1.json` |
| 3347 | GateMate n16 Ising tile build smoke | hardware | `results/experiment_3347_gatemate_n16_ising_tile_bitstream_build_smoke_v2.json` |
| 3348 | Reproducer pack and matrix v40 | aggregation | `results/experiment_3348_independent_reproducer_pack_evidence_matrix_v40.json` |
| 3349 | Capstone v309 | aggregation | `results/experiment_3349_capstone_v309.json` |

## Dependency Graph

```text
exp3337
  -> exp3338
      -> exp3339 (gate: runtime_receipt_clean == true)
          -> exp3340 (gate: energy_descent_bootstrap_ready == true)
      -> exp3344 (gate: runtime_receipt_clean == true)

exp3337
  -> exp3341
      -> exp3342
          -> exp3343 (gate: monitor_provenance_axis_ready == true)

exp3345 reads exp3334 and any clean .309 verifier outputs but is not hard-gated.
exp3346 and exp3347 are independent hardware continuity tracks.
exp3348 aggregates all available .309 artifacts.
exp3349 aggregates all available .309 artifacts and records blocked states honestly.
```

Structured conductor gates are used for:

- `exp3339`: `exp3338.runtime_receipt_clean == true`
- `exp3340`: `exp3339.energy_descent_bootstrap_ready == true`
- `exp3344`: `exp3338.runtime_receipt_clean == true`
- `exp3343`: `exp3342.monitor_provenance_axis_ready == true`

The capstone remains ungated so the milestone always emits a terminal artifact.

## Hardware Requirements

Live SOTA tasks require:

- Existing local cache for at least one mandated GGUF model, preferably the
  `cached_sota_pair(gpu_indices=(0, 1))` pair.
- Dual RTX 3090 CUDA visibility through `torch.cuda` or the repo's existing
  runtime helpers. Do not rely only on `nvidia-smi`.
- Tokenizer/runtime dependencies for Qwen/Gemma GGUF loading. If
  `sentencepiece`, `tiktoken`, `llama-cpp-python`, or tokenizer backend support
  is missing, write a blocked artifact with the exact import/load error.

Hardware tasks require:

- **KV260:** `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` must pass
  before board work. Use the SSH/xmutil/uio path; do not resurrect retired host
  SD-card preconditions.
- **GateMate A1:** use `yosys`, `nextpnr-himbaechel`, `gmpack`, and
  `openFPGALoader -c dirtyJtag --detect`. Do not use obsolete
  `nextpnr-gatemate`.
- Hardware artifacts must include command transcripts, tool versions, elapsed
  time, board identifiers when available, and precise blocked reasons.

## Evidence And Artifact Rules

Each experiment must emit a JSON artifact at the declared deliverable path with:

- `honest_verdict` starting with a terminal prefix such as `complete:`,
  `success:`, `passed:`, `shipped:`, or `blocked:`.
- `inference_substrate` matching the task substrate.
- `random_seed`, `duration_s`, `reproducibility_checksum`, and `files_updated`.
- Task-specific gates and metrics needed by downstream tasks.
- For live LLM tasks: `model_specs`, `gpu_status`, cache paths, and duration
  provenance.
- For hardware tasks: command transcript path or embedded transcript summary,
  toolchain versions, board preconditions, and blocked reasons.

Experiments that match prior failed or retired scopes must include
`prior_failures` with `retire_if_same_verdict: true` unless the task is a
standard recurring archive with a non-failure predecessor. No task may modify
`scripts/research_conductor.py`, and `research-roadmap.yaml` must remain
unchanged.

## Exit Criteria

Milestone `.309` succeeds if it leaves Carnot with:

- An honest `.308` archive and `.309` activation artifact.
- A clear SOTA GGUF runtime receipt that either unblocks live inference or
  precisely identifies the loader/tokenizer blocker.
- A runtime-clean energy-descent bootstrap and, if gated, a full
  energy-descent-vs-autoregressive SOTA panel.
- A verifier-diversity remediation plan plus reaudit after a new
  monitor/provenance axis.
- A continuous self-learning FR-11 v5 artifact with nonforgetting, rollback, and
  soundness/completeness metrics.
- Board-local KV260 and GateMate artifacts, either successful or honestly
  blocked.
- A reproducer/evidence matrix v40 and capstone v309 that record the next top
  gap without inflating claims.

## Non-Goals

- No public-doc marketing edits.
- No changes to `scripts/research_conductor.py`.
- No modification of `research-roadmap.yaml`.
- No headline claim from legacy small models, tiny smoke tests, or
  duration-flagged live inference.
- No greedy/diversity-maximizing verifier selection rerun matching retired
  exclusion-manifest scopes.
- No local TSU, Kona parity, or hardware speedup claim without measured local
  transcripts.

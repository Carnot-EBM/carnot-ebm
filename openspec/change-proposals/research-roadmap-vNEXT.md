# Research Roadmap vNEXT - Milestone 2026.05.303

**Title:** Prompt-Injection v4 Full Corpus + Garak Gate + Repair Reopen

**Created:** 2026-05-28
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.302
**Execution queue:** `exp3267` through `exp3280`

## What Milestone 2026.05.302 Proved

Milestone `.302` completed the reboot recovery path and moved the top blocker
from runtime bring-up to scientific evidence:

- `exp3260` archived `.301` and activated `.302`.
- `exp3261` confirmed CUDA recovery after the operator reboot: both RTX 3090s
  enumerated and selected-Python CUDA was usable.
- `exp3262` produced a llama.cpp CUDA receipt smoke.
- `exp3263` produced a mandated SOTA GGUF receipt with
  `sota_gguf_receipt_ready=true`; the cached mandated model was
  `unsloth/gemma-4-26B-A4B-it-GGUF`. The receipt was still flagged by
  methodology hygiene because the duration was too short for a clean headline
  evidence row.
- `exp3264` produced the first v4 prompt-injection teacher-label shard
  (`n=2000`, benign=1459, injection=541).
- `exp3265` trained/evaluated the v4 KAN sidecar on the shard
  (`n_train=1600`, `n_eval=400`, `shard_auroc=0.791096`). This is useful as
  a pilot, not a headline result.
- `exp3266` reported `paper_ready=false`, `publication_blocker_count=105`,
  `cuda_recovery_unblocked_sota_receipt=true`, and
  `next_top_gap=full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates`.

The natural next milestone is therefore not another CUDA repair milestone. It
is a corpus-scale evidence milestone: finish the v4 15k prompt-injection
corpus across shards, evaluate the KAN sidecar with real split and DeLong
controls, add Garak/adaptive red-team pressure, clean up SOTA receipt
methodology for downstream repair eligibility, and reopen repair only through
explicit gates.

## Three Biggest Gaps To PRD Vision

1. **Prompt-injection evidence is still a pilot shard, not a publication-grade
   verifier result.** The PRD vision requires verifiable reasoning and robust
   mitigation evidence. `.302` produced only a 2k shard and a non-headline KAN
   AUROC. Carnot still needs the full 15k corpus, leakage checks, class
   balance, aligned-instruction controls, adaptive attacks, DeLong
   non-inferiority, and Garak red-team gates.

2. **SOTA local evidence is available but not clean enough to reopen repair
   claims.** CUDA and llama.cpp are working, but the SOTA receipt has a
   duration/methodology flag and only one mandated SOTA model is currently
   cached. Repair and clean-verifier claims should not resume until a
   methodology supplement or long-duration receipt confirms the evidence row is
   headline-eligible.

3. **Continuous self-learning is still controller-memory evidence, not a
   retention-tested learning loop.** FR-11 requires autonomous self-learning
   without foundation-weight update claims. Recent memory benchmarks show that
   external memory can create negative transfer and forgetting. Carnot needs a
   held-out failure-memory audit over full-corpus prompt-injection failures and
   legacy gate-block traces before promoting the loop.

## External Research Integrated

The 2026-05-28 post-`.302` sweep was added to `research-references.md` before
this roadmap was designed. The most relevant updates are:

- Distributional EBMs (`https://arxiv.org/abs/2605.18871`) support a two-pass
  uncertainty/regeneration loop, but only as routing over exact constraints.
- ARM/EBM lookahead theory (`https://arxiv.org/abs/2512.15605`) clarifies
  verifier-to-generator distillation, but `.303` stays empirical and does not
  claim an ARM/EBM foundation-model result.
- Energy-based latent reasoning (`https://arxiv.org/abs/2603.28248`) reports a
  CNF failure mode from latent drift, reinforcing the need for drift and
  leakage audits before KAN promotion.
- AlignSentinel, DataFlip, and the 2026 prompt-injection survey
  (`https://openreview.net/forum?id=yPgbdOdOPG`,
  `https://arxiv.org/abs/2507.05630`,
  `https://arxiv.org/abs/2601.22240`) motivate aligned-instruction benign
  examples and adaptive-attack arms in the v4 corpus.
- Prompt-injection vulnerability work across Qwen/Gemma and ranker surfaces
  (`https://arxiv.org/abs/2602.22242`,
  `https://arxiv.org/abs/2602.16752`) motivates stratifying results by model
  family, task surface, and attack type.
- KAN cybersecurity papers (`https://arxiv.org/abs/2503.02281`,
  `https://arxiv.org/abs/2509.05259`) support KAN as an interpretable detector
  sidecar only when leakage and explanation controls exist.
- Agent-memory and continual-learning work (`https://openreview.net/forum?id=MSXbrNExax`,
  `https://arxiv.org/abs/2604.27003`, `https://arxiv.org/abs/2603.07670`,
  `https://arxiv.org/abs/2605.18421`, `https://arxiv.org/abs/2604.20087`)
  motivates FR-11 retention, transfer, and forgetting metrics.
- Garak (`https://github.com/NVIDIA/garak`) is the practical red-team gate for
  prompt injection and jailbreak probes. Extropic TSU and Logical Intelligence
  Kona remain strategic signals only; `.303` makes no hardware access or
  speedup claim.

## SOTA Local GGUF Policy

Any `.303` experiment that invokes an LLM for evidence must include at least
one mandated local SOTA GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as fast CPU smoke tests. They cannot
populate headline result fields and cannot unblock clean verifier, repair, or
publication-readiness claims.

## Architecture Diagram

```text
                 .302 terminal state
  CUDA recovered, llama.cpp receipt passed, one SOTA GGUF receipt landed,
  v4 prompt-injection shard n=2000, KAN shard AUROC=0.791096,
  paper_ready=false, blockers=105,
  next_top_gap=full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates
                              |
                              v
             exp3267 close .302 and open .303 corpus queue
                              |
                +-------------+--------------+
                |                            |
                v                            v
 exp3268 SOTA receipt methodology      exp3269 full-corpus
 supplement / clean eligibility        split manifest
                |                            |
                +-------------+--------------+
                              |
                              v
               exp3270 teacher-label shards 2-4
                              |
                              v
               exp3271 teacher-label shards 5-7
                    plus Garak/adaptive seed
                              |
                              v
               exp3272 full 15k assembly and leakage audit
                              |
                    +---------+----------+
                    |                    |
                    v                    v
          exp3273 KAN DeLong       exp3278 FR-11 retention,
          full-corpus eval          transfer, forgetting audit
                    |
                    v
          exp3274 Garak/DataFlip red-team gate

 exp3268 -> exp3275 clean SOTA verifier rerun
 exp3273 + exp3274 + exp3275 -> exp3276 repair gate decision
 exp3276.repair_gate_open -> exp3277 SOTA repair micro-panel

 exp3279 matrix v35 -> exp3280 capstone v303
```

## Phase Plan

### Phase 1 - Handoff and Evidence Hygiene

- `exp3267` closes `.302`, records the actual `.302` artifacts in the archive
  handoff, and activates the `.303` corpus-scale queue.
- `exp3268` turns the short `.302` SOTA receipt into a methodology supplement
  and, where feasible, a longer-duration receipt over cached mandated GGUFs. It
  emits `clean_sota_receipt_eligible` for downstream labels and repair.
- `exp3269` builds the v4 full-corpus split manifest: shard boundaries,
  sample-size ledger, aligned-instruction benign controls, adaptive attacks,
  constraint-tax arms, train/eval/holdout/Garak splits, and DeLong gates.

### Phase 2 - Full 15k Prompt-Injection Corpus

- `exp3270` labels shards 2-4 using a mandated local SOTA GGUF model and the
  `.302` shard as prior context.
- `exp3271` labels shards 5-7, adds the Garak/adaptive seed set, and reports
  cumulative label counts.
- `exp3272` assembles the full corpus, deduplicates, checks leakage, freezes
  train/eval/holdout/Garak splits, and emits `full_15k_corpus_ready`.

### Phase 3 - Detector, Garak, and Repair Reopen

- `exp3273` trains/evaluates the prompt-injection KAN sidecar on the full
  corpus and runs DeLong non-inferiority against exact and baseline detectors.
- `exp3274` applies Garak/DataFlip/adaptive red-team probes against the KAN and
  local SOTA target where available, recording pass/fail gates separately from
  AUROC.
- `exp3275` performs the clean local SOTA verifier rerun v14 only after the
  receipt methodology gate is clean.
- `exp3276` decides whether the repair gate can reopen based on full-corpus
  KAN evidence, Garak evidence, and clean verifier metrics.
- `exp3277` runs a small SOTA repair micro-panel only if `repair_gate_open` is
  true.

### Phase 4 - FR-11 and Aggregation

- `exp3278` is the required continuous self-learning experiment. It audits
  controller-memory retention, adaptation, forgetting, and negative transfer
  across full-corpus prompt-injection failures and older gate-block traces.
- `exp3279` builds evidence matrix v35 with shard, KAN, Garak, clean verifier,
  repair, and FR-11 fields.
- `exp3280` produces the `.303` capstone and names the next top gap.

## Dependency Graph

```text
exp3267
  -> exp3268
  -> exp3269

exp3268.clean_sota_receipt_eligible
  -> exp3270
  -> exp3275

exp3269.full_corpus_manifest_ready
  -> exp3270
      -> exp3271 [gate: cumulative_label_count >= 8000]
          -> exp3272 [gate: cumulative_label_count >= 14000
                             and garak_seed_count >= 1000]
              -> exp3273 [gate: full_15k_corpus_ready == true]
                  -> exp3274 [gate: v4_full_eval_ready == true]
              -> exp3278 [gate: full_15k_corpus_ready == true]

exp3273.v4_full_eval_ready + exp3274.garak_redteam_eval_ready
  + exp3275.clean_verifier_rerun_ready
  -> exp3276
      -> exp3277 [gate: repair_gate_open == true]

exp3279 reads all available .303 artifacts
exp3280 reads exp3279 and all available .303 artifacts
```

## Hardware Requirements

- **Required for live LLM tasks:** at least one visible NVIDIA GPU with CUDA,
  llama.cpp CUDA/offload support, and at least one cached or resolvable
  mandated SOTA GGUF. Every live LLM task must record `MODEL_SPECS`,
  `preconditions_checked`, `models_used`, GPU memory evidence, duration, and
  cache status.
- **Recommended for label throughput:** dual RTX 3090 availability. If only
  one GPU is usable, shard tasks should reduce batch size and record throughput
  rather than failing silently.
- **Allowed CPU-only tasks:** corpus assembly, leakage/de-dup audits, DeLong
  statistics after labels exist, FR-11 memory audits, matrix, and capstone.
- **Not required for `.303`:** KV260, GateMate, PolarFire, Extropic TSU, or
  Kona hardware. `.303` must not claim TSU/Kona access, thermodynamic speedup,
  or FPGA acceleration. Do not use host `/dev/mmcblk*` KV260 preconditions.

## Experiment Queue

1. `exp3267-close-v302-open-v303-corpus-queue`
2. `exp3268-sota-receipt-methodology-supplement-v1`
3. `exp3269-prompt-injection-v4-full-corpus-split-manifest-v1`
4. `exp3270-prompt-injection-teacher-label-shards-2-4-v1`
5. `exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1`
6. `exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1`
7. `exp3273-prompt-injection-kan-full-corpus-delong-eval-v1`
8. `exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1`
9. `exp3275-clean-local-sota-verifier-rerun-v14`
10. `exp3276-repair-gate-decision-v8-after-v4-garak-clean-verifier`
11. `exp3277-sota-repair-micro-panel-v9`
12. `exp3278-fr11-full-corpus-continual-self-learning-audit-v1`
13. `exp3279-evidence-matrix-v35`
14. `exp3280-capstone-v303`

## Done Criteria

- `research-roadmap-next.yaml` validates against roadmap schema,
  prior-failure discipline, exclusion-manifest lint, and gate audit.
- Every live-LLM task includes `MODEL_SPECS` with at least one mandated SOTA
  GGUF model.
- Every gated task includes matching `gated_on` metadata and the upstream
  prompt lists the gated artifact field under `REQUIRED ARTIFACT FIELDS`.
- The prompt-injection v4 corpus reaches the 15k target through split shards or
  honestly reports the precise blocker and remaining count.
- KAN claims remain sidecar-only unless full-corpus DeLong and Garak gates pass.
- Repair runs only through `repair_gate_open == true`.
- At least one experiment (`exp3278`) directly advances continuous
  self-learning under FR-11 without claiming foundation-model weight updates.
- No task modifies `scripts/research_conductor.py`, no task modifies
  `research-roadmap.yaml`, and no task pushes.

# Research Roadmap v42 — Milestone 2026.04.42

**Status:** Proposed
**Milestone:** 2026.04.42
**Title:** Break the Synthetic Barrier — Live Data Sprint, Root Cause Diagnosed, JEPA Recovery
**Planned Experiments:** 549–562 (14 experiments)
**Planned Date:** 2026-04-20 onwards

---

## What Milestone 2026.04.41 Proved

Milestone .41 confirmed five important facts while opening four new critical RETROs:

1. **Live GPU inference is no longer blocked.** Exp 538 ran 25 questions with
   `inference_mode='live_gpu'`, `status='success'`. RETRO-053 (env_autofix falsy gate)
   and RETRO-055 (latency budget) are both resolved. The pipeline physically reaches the GPU.

2. **No improvement detected on live data (RETRO-033 miss #10).** `live_no_improvement_25q`.
   Root cause is unknown: it could be high FP rate from VeriCoT/VPRM on IT-model responses,
   or repair generating incorrect fixes, or re-verification logic. Diagnostic work is needed
   before another raw benchmark attempt.

3. **Synthetic proxy epidemic is the #1 blocker (RETRO-058 CRITICAL).** Six of eleven
   experiments fell back to synthetic or proxy data: Exps 540, 542, 543, 545, 546, and
   the GRPO retrain (n_pairs=3). Root cause: FOVER corpus has only 24 diverse pairs
   (88% carry violations, 12% other). Every model retrain failed because of data starvation,
   not code or GPU issues.

4. **JEPA trained on 24 pairs is anti-correlated (RETRO-056).** Exp 543 produced
   `final_auc=0.444` — below the 0.5 random baseline. The predictor learned the WRONG
   direction. Fix: grow FOVER corpus to >=100 diverse pairs with constraint_type_entropy >= 1.5
   bits BEFORE any JEPA retrain.

5. **LowRankKAEM energy accuracy outside tolerance (RETRO-057).** Exp 544 wired LowRankKAEM
   as default (4-155x speedup confirmed) but `energy_mad_normalized ≈ 0.96-0.99` — far outside
   the 5% production tolerance. Fix: SVD rank sweep + calibration layer.

6. **Conductor exclusion manifest still missing (RETRO-059).** Exp 547 confirmed that
   Exps 308, 260, 309, 425, 410 ARE already fully modern. Without an exclusion manifest,
   the conductor will re-select them for infrastructure sweeps in every future milestone.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: Synthetic Proxy Epidemic Blocks All Model Learning (FR-11, RETRO-058)

**This is the root cause of multiple failures.** The entire Tier 1-3 self-learning pipeline
depends on real (question, response, is_correct) pairs from live inference. With only 24 pairs
(88% carry violations), the corpus is too small and too skewed to train any model. EORM, JEPA,
InternalStateProbe, and AutoRefine templates all fell back to synthetic in .41.

**Fix in Exps 551-553:** Dedicated live data collection sprint. Run 50+50 questions through both
models (Gemma4-E4B-it and Qwen3.5-0.8B), NO repair pipeline during collection (just inference
and FOVER annotation). Merge with existing pairs from Exps 442 and 538. Target: >=100 real pairs
with `constraint_type_entropy >= 1.5 bits`. Only then schedule model retraining.

This is the gating condition for Exps 556, 557, 558, and 561.

### Gap 2: Verify-Repair Root Cause Unknown (RETRO-033, miss #10)

**We know the pipeline reaches the GPU. We do NOT know why it produces no improvement.**
Three hypotheses, in descending likelihood:

1. **High FP rate:** VeriCoT/VPRM finds violations in responses that are actually correct
   (IT-format responses don't match regex patterns). Repair then corrupts a correct answer.
2. **Poor repair quality:** Violations correctly detected, but the repair generation produces
   an incorrect or ill-formed fix.
3. **Re-verification logic:** Repaired responses pass verification but the evaluation
   comparator doesn't detect the improvement (scoring bug).

**Fix in Exps 554-555:** Load Exp 538's 25 live responses and their known labels. Run VeriCoT
and VPRM separately. For known-correct responses, measure FP rate per extractor. For
known-incorrect responses, measure TP rate. This diagnostic tells us EXACTLY where the
pipeline fails. Then implement confidence-weighted filtering to reduce FP.

### Gap 3: JEPA Trained Backwards on Biased Corpus (RETRO-056)

**JEPA v8 (Exp 543) is actively harmful** — AUC=0.444 means it predicts violations are
LESS likely when they actually are MORE likely. The 24-pair corpus with 88% carry violations
caused the model to over-fit carry-check patterns and generalize poorly. This is a data
quality problem, not a model problem.

**Fix in Exps 553 + 557:** First, build a diverse corpus (Exp 553). Then retrain JEPA v9
on that corpus using the LeWorldModel two-term objective (prediction + Gaussian KL
regularization from arXiv 2603.19312), which provides stable training without curriculum
scheduling. Target: AUC >= 0.800 with stable training from epoch 1.

---

## Architecture Diagram (Verification Cascade — .42 State)

```
LLM Response
    │
    ▼
[Tier 0a] CarnotThinkProbe (generative 3-step CoT verdict)
    │  verdict='incorrect' → fast-path violation
    ▼
[Tier 0b] SpilledEnergyDetector (~0ms, logit-discrepancy)
    │  high_spill_fraction → likely hallucination
    ▼
[Tier 0c] NUP Probe v4 (~0ms, contrastive bigram energy, AUC=1.0)
    │  low energy → likely correct, skip downstream
    ▼
[Tier 0d] HallucinationBasinDetector (~0ms, latent basin depth)
    │  deep basin → stable reasoning, skip downstream
    ▼
[Tier 1] SinkProbe (~0ms, attention sink concentration)
    │  high sink score → skip downstream
    ▼
[Tier 2] EORM / InternalStateProbe (~10ms / <1ms)
    │  .42: EORM retrained on 100+ real pairs (Exp 556)
    │  .42: InternalStateProbe trained on real data (Exp 558)
    │  energy < threshold → skip Ising
    ▼
[Tier 3] VerifyRepairPipeline
    │  .42: Confidence-weighted extraction (Exp 555) — reduces FP
    │  .42: LatentCoTEBMCalibrator (Exp 560) — step-level energy guidance
    │  violations found → repair → re-verify
    ▼
[Self-Learning] ConstraintAdditionFromMemory
    │  .42: Wired to real FP patterns from Exp 554 (Exp 561)
    ▼
[FR-11] JEPA Predictor (predicts violations before generation)
    │  .42: JEPA v9 retrained on 100+ diverse pairs (Exp 557)
    │  .42: LowRankKAEM calibrated (Exp 559)
    ▼
Verified Output
```

---

## Phase Descriptions

### Phase 1: Infrastructure — Exclusion Manifest + Zombie Kill Fix (Exps 549–550)

**Goal:** Prevent conductor from re-selecting already-modern experiments.
Fix kill_gpu_zombies() to work without pynvml installed.

Exp 549 implements `scripts/conductor_exclusion_manifest.json` with Exps 308, 260, 309, 425, 410
and adds `ExperimentTemplate.check_exclusion_manifest()` — if an experiment ID is in the manifest,
the script exits immediately with an "excluded" artifact (< 1 second). Also fixes
`kill_gpu_zombies()` to use nvidia-smi subprocess fallback when pynvml is not importable.
Kills zombie PIDs 527256, 527259, 529495 from .41 close.

Exp 550 (Exp 547b) actually migrates the 5 slowest scripts to use `BatchedInferenceRunner`
inner loops. Exp 547 confirmed these scripts are "modern" (ExperimentTemplate, watchdog, teardown)
but `batching_added=[]` — they still run sequential inference loops. This is the
8.5% wall-time savings the retro projected but didn't deliver.

### Phase 2: Live Data Collection Sprint (Exps 551–553)

**Goal:** Grow real CoT corpus from 24 to >=100 diverse pairs before ANY model retraining.
This phase is the gating condition for all Phase 4 experiments.

Exps 551 and 552 run 50 questions each (GSM8K seed=42, indices 0-49 and 50-99) through both
Gemma4-E4B-it and Qwen3.5-0.8B. No repair pipeline — just inference and FOVER annotation.
Each question yields 2 CoT chains (one per model); FOVER labels each as correct/incorrect.
Target: 100+ labeled CoT steps per batch, 200+ total.

Exp 553 merges all available real pairs (Exps 442, 538, 551, 552), runs constraint-type
diversity audit (constraint_type_entropy target >= 1.5 bits), and filters/balances to
ensure the corpus is not dominated by carry violations. Writes `results/fover_corpus_v2.json`.

### Phase 3: Root Cause Analysis — Why No Improvement? (Exps 554–555)

**Goal:** Diagnose exactly why Exp 538 showed `live_no_improvement_25q`.

Exp 554 runs VeriCoT and VPRM separately on Exp 538's 25 live responses with known
labels. Reports per-extractor FP rate (on correct responses) and TP rate (on incorrect
responses). This is the diagnostic that tells us whether the problem is extraction,
repair, or re-verification.

Exp 555 implements `ConfidenceWeightedConstraint`: each violation gets a confidence score
(0-1) based on how definitively the arithmetic check fails. Only violations above the
threshold trigger repair. Tests thresholds [0.5, 0.7, 0.9] on Exp 538's 25 questions.
Target: reduce FP by 50%+ without losing >20% TP.

### Phase 4: Model Retraining on Real Data (Exps 556–558)

**Goal:** Retrain EORM, JEPA, and InternalStateProbe on the 100+ real pairs from Phase 2.
These experiments are gated on Exp 553 (fover_corpus_v2.json exists).

Exp 556 retrains EORM using GRPO contrastive pairing (arXiv 2503.06639): real
(correct, incorrect) response pairs from fover_corpus_v2 form the contrastive signal
without additional labeling. Target: AUC >= 0.700 with `inference_mode='real_data'`.

Exp 557 retrains JEPA v9 on the diverse 100+ corpus using LeWorldModel's two-term
objective (arXiv 2603.19312): L = L_prediction + λ * KL(q || N(0,I)). Filters corpus
to ensure `constraint_type_entropy >= 1.0` in training data. RETRO-056 closure attempt.
Target: AUC >= 0.800, stable training from epoch 1.

Exp 558 trains InternalStateProbe (arXiv 2511.06209) on real data (80/20 split from
fover_corpus_v2). The probe is a single linear layer on the LLM hidden state at layer -4.
Compare AUC vs EORM (55M params). If AUC >= 0.700 at 810x smaller, promote to Tier 2 default.

### Phase 5: New Research (Exps 559–560)

**Goal:** Close RETRO-057 (LowRankKAEM) and test the filed .42 paper (arXiv 2511.07124).

Exp 559 implements a calibration layer for LowRankKAEMEnergy to close RETRO-057.
SVD rank sweep k=2,4,8,16,32; for each k, trains an affine calibration layer on 1000
synthetic Ising instances. Target: find k where `energy_mad_normalized < 0.05` AND
`speedup > 5x`. Tune the default k for production use.

Exp 560 implements `LatentCoTEBMCalibrator` (arXiv 2511.07124): wraps the generation
loop, computes EORM energy at each 32-token boundary, applies soft temperature adjustment
(`logits *= (1 - alpha * energy_score)`) toward lower-energy continuations. Tests on 25
synthetic GSM8K questions. Compare violation rate vs uncalibrated baseline.

### Phase 6: FR-11 Self-Learning + Retrospective (Exps 561–562)

**Goal:** Complete the FR-11 self-learning relay on real data. Close out the milestone.

Exp 561 (FR-11 mandatory): Runs Tier 1 self-learning relay using real FP patterns from
Exp 554. Session 1: verify 25 questions from Exp 538 with base constraints. Analyze
FP patterns. `ConstraintAdditionFromMemory` adds new constraints from real FP patterns.
Session 2: verify same 25 questions with extended constraints. Measure FP rate delta.
Must use `inference_mode='real_data'`. Closes the live-data→learning loop.

Exp 562: Standard milestone retrospective (v16 schema). Evaluates all success criteria,
opens new RETRO items, measures wall-time efficiency improvement.

---

## Dependency Graph

```
Exp 549 (exclusion manifest + zombie kill)
    └── Exp 550 (BatchedInferenceRunner real migration)

Exp 551 (live 50q data A — GPU)
    └── Exp 552 (live 50q data B — GPU)
            └── Exp 553 (FOVER corpus v2 — merge + diversity)
                    ├── Exp 556 (EORM GRPO retrain on real)
                    ├── Exp 557 (JEPA v9 retrain — RETRO-056)
                    ├── Exp 558 (InternalStateProbe real data)
                    └── Exp 561 (FR-11 Tier 1 relay — real data)

Exp 554 (extraction diagnostic on Exp 538 live responses)
    ├── Exp 555 (confidence-weighted constraint filtering)
    └── Exp 561 (FR-11 Tier 1 relay uses FP analysis from 554)

Exp 559 (LowRankKAEM calibration — RETRO-057)    [independent]
Exp 560 (LatentCoTEBMCalibrator — arXiv 2511.07124) [independent]

Exp 562 (retrospective — reads all above)
```

---

## Success Criteria

| Criterion | Target | Experiment |
|-----------|--------|------------|
| `zombie_kill_working` | `killed_pids != []` OR `nvidia_smi_fallback=True` | Exp 549 |
| `exclusion_manifest_created` | True | Exp 549 |
| `retro_059_resolved` | True | Exp 549 |
| `batching_added` | All 5 scripts | Exp 550 |
| `live_50q_a_completed` | `inference_mode=live_gpu` | Exp 551 |
| `live_50q_b_completed` | `inference_mode=live_gpu` | Exp 552 |
| `fover_corpus_v2_n_labeled` | >= 100 | Exp 553 |
| `fover_corpus_v2_entropy` | >= 1.5 bits | Exp 553 |
| `extraction_fp_rate_qwen` | Measured (any value) | Exp 554 |
| `confidence_threshold_reduces_fp` | delta_fp < -0.3 | Exp 555 |
| `eorm_real_data_auc` | >= 0.700 | Exp 556 |
| `jepa_v9_auc` | >= 0.800 | Exp 557 |
| `retro_056_closed` | True | Exp 557 |
| `internal_probe_real_auc` | >= 0.700 | Exp 558 |
| `lowrank_calibrated_mad` | < 0.05 | Exp 559 |
| `retro_057_closed` | True | Exp 559 |
| `latent_cot_violation_delta` | < 0 (fewer violations) | Exp 560 |
| `fr11_real_data_relay` | `inference_mode=real_data` | Exp 561 |
| `fp_rate_delta` | < 0 | Exp 561 |

---

## Hardware Requirements

| Experiment | GPU Required | VRAM | Model |
|------------|-------------|------|-------|
| Exp 549 | No | — | CPU infrastructure |
| Exp 550 | No | — | CPU script migration |
| Exp 551 | Yes (CRITICAL) | 10+1.5 GiB | Gemma4-INT4 + Qwen3.5-0.8B |
| Exp 552 | Yes (CRITICAL) | 10+1.5 GiB | Gemma4-INT4 + Qwen3.5-0.8B |
| Exp 553 | No | — | CPU FOVER merge |
| Exp 554 | No | — | CPU diagnostic |
| Exp 555 | No | — | CPU filtering |
| Exp 556 | No | — | CPU EORM retrain |
| Exp 557 | No | — | CPU JEPA retrain |
| Exp 558 | No | — | CPU probe training |
| Exp 559 | No | — | CPU KAEM calibration |
| Exp 560 | No | — | CPU synthetic inference |
| Exp 561 | No | — | CPU self-learning relay |
| Exp 562 | No | — | CPU retrospective |

**FIRST ACTION in every experiment:** Kill zombie PIDs from .41 close.

```bash
kill -9 527256 527259 529495 2>/dev/null; sleep 2; nvidia-smi
```

---

## Open RETRO Items Addressed

| RETRO | Description | Addressed By |
|-------|-------------|-------------|
| RETRO-033 | Live verify-repair positive result (miss #10) | Exps 554-555 (diagnose + fix FP) |
| RETRO-038 | Live 200q VeriCoT+VPRM statistically significant | Exp 552 (data collection) + .43 |
| RETRO-052 | GPU1 0% compute utilization | Exps 551-552 (DualGPU explicit routing) |
| RETRO-056 | JEPA AUC=0.444 below random on 24-pair corpus | Exps 553 + 557 |
| RETRO-057 | LowRankKAEM energy outside 5% tolerance | Exp 559 |
| RETRO-058 | Synthetic proxy epidemic — 6/11 experiments | Exps 551-553 (data sprint) |
| RETRO-059 | Conductor exclusion manifest not created | Exp 549 |

---

## arxiv Findings Incorporated

| Paper | What | Incorporated As |
|-------|------|----------------|
| arXiv 2511.07124 | EBM calibration of latent CoT reasoning (filed .42) | Exp 560 |
| arXiv 2503.06639 | GRPO contrastive pairing for verifiable rewards | Exp 556 (EORM retrain) |
| arXiv 2603.19312 | LeWorldModel two-term JEPA objective | Exp 557 (JEPA v9) |
| arXiv 2511.06209 | Internal-state probes (810x smaller than PRM) | Exp 558 (real data redo) |
| arXiv 2603.20224 | Energy-per-Token metric for LLM inference efficiency | New reference (future) |

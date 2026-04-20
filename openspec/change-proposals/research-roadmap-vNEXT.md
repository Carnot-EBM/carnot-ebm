# Research Roadmap v46 — Milestone 2026.04.46

**Status:** Proposed
**Milestone:** 2026.04.46
**Title:** Live Distribution Fix — Data-Driven Extractor Retraining and JEPA Generalization
**Planned Experiments:** 601–613 (13 experiments)
**Planned Date:** 2026-04-20 onwards
**Supersedes:** Milestone 2026.04.45 — "Live-Calibrated CoACE and DSVD — Closing the Offline/Live Distribution Gap"

---

## What Milestone 2026.04.45 Proved

Milestone .45 produced one clear validation and two new critical failures:

1. **JEPA v12 AUC=1.0 on 100-pair live corpus (Exp 593, RETRO-063 validated).**
   CPMI contrastive pairing + PROGRS outcome-centering trained JEPA v12 on the full Exp 578
   live corpus (100 real pairs) and achieved AUC=1.0 on the in-distribution validation split.
   CAVEAT: AUC=1.0 on a 20-pair held-out split from the same session is almost certainly
   overfitting. The architecture is sound; the generalization must be validated on held-out
   questions never seen in training (GSM8K 500+).

2. **CoACEV3 recall REGRESSED to 4% on live data — WORSE than v2's 5.9% (RETRO-068).**
   Exp 591 tested CoACEExtractorV3 (narrative + percentage + unit conversion patterns) on 25
   live production responses. Result: v3_recall=0.04, gate_open=False. The v3 architecture
   added prose patterns but the patterns were still engineered from intuition, not from analysis
   of real live model outputs. The offline/live distribution gap widened despite more patterns.
   Root cause diagnosis confirmed: NO hand-engineered pattern set will bridge this gap.
   Only DATA-DRIVEN calibration on actual live IT-model outputs can fix this.

3. **DSVD live AUC=0.586 vs 0.976 offline — same distribution mismatch (RETRO-069).**
   Exp 592 validated DSVDAdapter on live pairs. dsvd_live_auc=0.586 is well below the 0.80
   deployment threshold. The same root cause as CoACE: the detector was trained/calibrated
   on synthetic hidden-state patterns that don't match live Qwen3.5-0.8B or Gemma4-E4B-it
   inference. Fine-tuning on the Exp 578 live corpus is the required fix.

4. **D-Wave cloud access confirmed (Exp 598): speedup_ratio=26.24x over CPU Neal.**
   DWaveNealBackend is available and faster. Ready to wire into production SamplerBackend.

5. **Exclusion manifest built but conductor_consulted=False (RETRO-067 partial).**
   Same 5 experiments appeared in slowest-5 for NINTH consecutive milestone.
   Cumulative waste since .37: ~3,255 minutes (54.3 hours). Manifest must be WIRED in .46.

6. **NUP Probe v5 AUC=0.739 (Exp 599) — below 0.80 Tier 0c deployment threshold (RETRO-049).**
   GRPO contrastive pairing improved v4 (synthetic AUC=1.0) to a more realistic v5 score.
   Needs retraining on larger live corpus to cross the 0.80 deployment threshold.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: Offline/Live Distribution Mismatch — CRITICAL BLOCKER (RETRO-066, 068, 069, 033)

**Problem:** Both extraction approaches (CoACE post-hoc and DSVD mid-generation) fail on live data.
- CoACEV3: 86.7% recall offline → 4% recall on live IT-model outputs
- DSVD: 0.976 AUC offline → 0.586 AUC on live IT-model outputs

**Root cause confirmed:** The training/calibration data for both extractors is synthetic or
curated offline. Real IT-model error patterns are fundamentally different:
- Narrative arithmetic: "Adding 47 to 28 gives us 76" (no `=` sign)
- Multi-paragraph derivations: error 3 steps before the stated conclusion
- Implicit quantity chains: "the subtotal is now X" without explicit equation
- Domain-specific vocabulary: percentage / ratio / word problems

**The fix:** Analyze the actual incorrect responses in the 100 live pairs from Exp 578.
Build CoACEV4 from frequency analysis of real errors, not from pattern intuition.
Fine-tune DSVDAdapter on real live hidden-state pairs, not synthetic stubs.

**Why PRD cares:** FR-12 (Verifiable Reasoning) — at 4% live recall, Carnot identifies
essentially ZERO real violations. Until this is fixed, RETRO-033 (live verify-repair positive)
cannot be resolved. This is the single highest-leverage action for the entire project.

### Gap 2: JEPA v12 Generalization Unknown (FR-11, RETRO-063)

**Problem:** JEPA v12 AUC=1.0 on in-distribution training split suggests overfitting.
The 100-pair Exp 578 corpus covers only GSM8K questions 0-49. A real generalization
test requires held-out questions from a different index range (GSM8K 500+) that were
never seen during training.

**Additionally:** Corpus diversity is limited. All pairs come from the same question set
(GSM8K 0-49) evaluated at one session. More diverse data (different question types,
longer chains, HumanEval code questions) would produce a more robust predictor.

**The fix:** Collect more live pairs (GSM8K 250-349) and validate JEPA v12 on strictly
held-out questions (GSM8K 500+). If AUC drops below 0.70, retrain JEPA v13 on the
expanded corpus.

### Gap 3: Exclusion Manifest Not Wired — 54 Hours Wasted (RETRO-067)

**Problem:** The exclusion manifest (scripts/conductor_exclusion_manifest.json) was built
in Exp 575 and wiring scripts created in Exp 589, but conductor_consulted=False in the
.45 retrospective. The same 5 experiments (308, 260, 309, 425, 410) consumed ~385 min
in .45 alone — experiments already modernized that keep re-entering because there is no
mechanism to prevent it.

**The fix:** Create a verifiable conductor session pre-check that reads the manifest at the
START of every session before any experiment is queued. Document the exact invocation
pattern the human conductor operator must use to verify the manifest is working.

---

## Verification Pipeline Architecture (Current State → Target for .46)

```
Input LLM Response
       │
    ┌──▼──────────────────────────────┐
    │  Tier 0a: CarnotThinkProbe      │ ~0ms (CI stub) / 50-200ms (GPU)
    │  Generative CoT verdict          │ → skip if 'incorrect'
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 0b: SpilledEnergyDetector │ ~0ms
    │  Per-token logit discrepancy     │
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 0c: NUP Probe v5          │ ~0ms (bigram dot product)
    │  AUC=0.739 [TARGET: v6 >=0.80]  │ ← .46 fix: retrain on larger corpus
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 0d: HallucinationBasin    │ ~0ms
    │  Latent-space basin depth        │
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 0e: HalluField            │ ~1ms (advisory only)
    │  Thermodynamic instability       │
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 1: SinkProbe              │ ~0ms (attention sink)
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 2: EORM                   │ ~10ms
    │  55M param CoT energy reward     │
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 2.5: DSVDAdapter          │ ~5ms
    │  AUC=0.586 live                 │ ← .46 fix: fine-tune on live corpus
    │  [TARGET: >=0.80 for deployment] │
    └──────────────────────┬──────────┘
                           │ (pass)
    ┌──────────────────────▼──────────┐
    │  Tier 3: CoACEExtractorV4       │ ~1ms
    │  recall=4% live (RETRO-068)     │ ← .46 fix: data-driven retraining
    │  [TARGET: >=20% for VR gate]    │
    └──────────────────────┬──────────┘
                           │ (violation)
    ┌──────────────────────▼──────────┐
    │  Ising Repair + HISR/D-Wave     │ 0.006ms/constraint (CPU)
    │  ConstraintAdditionFromMemory   │   or 26x faster via D-Wave
    └─────────────────────────────────┘
```

---

## Phase Structure

### Phase 0: Infrastructure First (Exps 601-602)

**Objective:** Prevent the same 5 legacy experiments from wasting another ~385 min in .46.
Expand the live corpus before any new training.

| Experiment | Title | Required? | GPU? |
|-----------|-------|-----------|------|
| 601 | ExclusionManifest Final Verification | Yes | No |
| 602 | Live Corpus Expansion v2 (100 more pairs) | Yes | YES |

### Phase 1: Fix Offline/Live Distribution Gap (Exps 603-606)

**Objective:** RETRO-068 and RETRO-069 — data-driven retraining of CoACEV4 and DSVD on
actual live model outputs. This is the CRITICAL PATH for RETRO-033.

| Experiment | Title | Required? | GPU? |
|-----------|-------|-----------|------|
| 603 | CoACEV4 Data-Driven Live Training | Critical | No |
| 604 | DSVD Live Fine-Tuning | Critical | No |
| 605 | Live Extractor Diagnostic v4 (gate) | Critical | No |
| 606 | Interleaved Formal Logic Verification (arXiv 2601.22642) | Research | No |

### Phase 2: JEPA Generalization + NUP Probe v6 (Exps 607-608)

**Objective:** Validate JEPA v12 on truly held-out questions. Retrain NUP Probe v6
to cross the 0.80 deployment threshold.

| Experiment | Title | Required? | GPU? |
|-----------|-------|-----------|------|
| 607 | JEPA v12 OOD Validation + Corpus v4 | FR-11 | No |
| 608 | NUP Probe v6 Live Retrain | Research | No |

### Phase 3: Live Verify-Repair (Exp 609)

**Objective:** RETRO-033 Attempt #14 — GATED on Exp 605 recall >= 0.20.

| Experiment | Title | Required? | GPU? |
|-----------|-------|-----------|------|
| 609 | Live VR CoACEV4 (RETRO-033 #14) | GATED | YES |

### Phase 4: Hardware + New Research (Exps 610-612)

**Objective:** Wire D-Wave into production. Implement FLIP reward calibration for
self-learning. FACT-E causal faithfulness + p-bit Ising for FPGA path.

| Experiment | Title | Required? | GPU? |
|-----------|-------|-----------|------|
| 610 | D-Wave Backend Wire-In + HISR Integration | Research | No |
| 611 | FLIP Reward Calibration + FR-11 Real Violations v5 | FR-11 | No |
| 612 | FACT-E Causal Faithfulness + Synchronous p-bit Ising RTL | Research | No |

### Phase 5: Retrospective (Exp 613)

| Experiment | Title | Required? | GPU? |
|-----------|-------|-----------|------|
| 613 | Milestone 2026.04.46 Operational Retrospective | Yes | No |

---

## Dependency Graph

```
Exp 601 (ExclusionManifest Wire-In)
    │
    ├──→ Exp 602 (Live Corpus Expansion v2)     [GPU REQUIRED]
    │         │
    │         ├──→ Exp 603 (CoACEV4 Data-Driven Training)
    │         │         │
    │         │         └──→ Exp 605 (Extractor Diagnostic v4)
    │         │                   │
    │         │                   ├── [gate: recall >= 0.20] ──→ Exp 609 (Live VR CoACEV4)
    │         │                   │                                   │
    │         │                   │                                   └──→ Exp 611 (FR-11 v5)
    │         │                   └── [always] ──→ Exp 613 (Retro)
    │         │
    │         ├──→ Exp 604 (DSVD Live Fine-Tuning)
    │         │         │
    │         │         └──→ Exp 605 (Extractor Diagnostic v4)  [same gate]
    │         │
    │         └──→ Exp 607 (JEPA v12 OOD Validation)  [FR-11, no gate]
    │
    ├──→ Exp 606 (Interleaved Formal Logic)      [independent research]
    ├──→ Exp 608 (NUP Probe v6)                  [independent research]
    ├──→ Exp 610 (D-Wave Wire-In + HISR)         [independent research]
    └──→ Exp 612 (FACT-E + Synchronous Ising)    [independent research]
```

---

## Success Criteria

| Criterion | Target | Experiment |
|-----------|--------|-----------|
| retro_067_verified | conductor_consulted=True logged | Exp 601 |
| live_corpus_v2_collected | n_new_pairs >= 80 | Exp 602 |
| coace_v4_live_recall | >= 0.20 | Exp 605 |
| dsvd_live_auc_improved | >= 0.80 | Exp 605 |
| retro_033_resolved | signed_improvement > 0 AND live_gpu | Exp 609 |
| jepa_v12_ood_auc | >= 0.65 (not overfit) | Exp 607 |
| nup_v6_auc | >= 0.80 | Exp 608 |
| dwave_wired | CARNOT_SAMPLER=dwave works | Exp 610 |
| fr11_relay | fr11_real_violations_confirmed=True | Exp 611 |
| fact_e_viable | causal_gap > 0 (faithfulness signal present) | Exp 612 |

---

## Open RETROs Addressed

| RETRO | Priority | Address In |
|-------|----------|-----------|
| RETRO-033 (13+ misses — live VR precision) | CRITICAL | Exp 609 (gated) |
| RETRO-038 (200q Wilson CI) | CRITICAL | BLOCKED until RETRO-033 |
| RETRO-049 (NUP Probe below threshold) | HIGH | Exp 608 |
| RETRO-057 (LowRankKAEM tolerance) | LOW | defer to .47 |
| RETRO-064 (CoACE live recall) | CRITICAL | Exp 603 |
| RETRO-065 (RAPL unavailable) | LOW | defer |
| RETRO-066 (distribution gap — v3 worse) | CRITICAL | Exp 603 |
| RETRO-067 (manifest not wired) | HIGH | Exp 601 |
| RETRO-068 (CoACEV3 4% live recall) | CRITICAL | Exp 603 |
| RETRO-069 (DSVD 0.586 live AUC) | HIGH | Exp 604 |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 602 | RTX 3090 (CUDA) | CARNOT_FORCE_LIVE=1 mandatory |
| Exp 609 | RTX 3090 x2 (CUDA) | GPU REQUIRED; gate must be open |
| All others | CPU only | JAX_PLATFORMS=cpu |
| KV260 FPGA | Vivado synthesis pending | Human must install Vivado 2023.2 |
| AMD XDNA NPU | Blocked | Human must: sudo pacman -S ninja openblas |

---

## ArXiv References Incorporated

| Paper | What | Experiment |
|-------|------|-----------|
| arXiv 2504.00891 | GenPRM: generative CoT extractor — LLM-as-extractor for CoACEV4 | Exp 603 |
| arXiv 2601.02170 | Streaming Hallucination Detection — temporal state tracking for DSVD fine-tuning | Exp 604 |
| arXiv 2604.12632 | CAPO: calibration-aware contrastive loss for NUP Probe v6 | Exp 608 |
| arXiv 2601.22642 | Interleaved Formal-Logic Verification during generation | Exp 606 |
| arXiv 2604.10693 | FACT-E: causal faithfulness for CoT | Exp 612 |
| arXiv 2604.01564 | Synchronous p-bit Ising machine (50% FPGA area reduction) | Exp 612 |
| arXiv 2602.13551 | FLIP: backward inference reward calibration | Exp 611 |
| arXiv 2603.18683 | HISR: hindsight credit assignment | Exp 610 |
| arXiv 2601.00095 | MetaJuLS: meta-RL constraint propagation (filed for .47) | Future |

---

## Milestone Question

**Has RETRO-033 been resolved?** After 13 failed attempts, does data-driven extractor retraining
on live IT-model outputs finally unlock the first live positive verify-repair result?

**Success:** signed_improvement > 0 with inference_mode='live_gpu'.
**Failure criterion:** If Exp 605 gate stays closed (CoACEV4 recall < 0.20), do NOT schedule
Exp 609. The lesson from .45 is: never schedule VR unless recall >= 20%. Escalate to RETRO-070
and redesign extraction from scratch (LLM-as-extractor approach per research-program.md Goal #1b).

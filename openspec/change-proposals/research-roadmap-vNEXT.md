# Research Roadmap — Milestone 2026.04.55

**Title:** JEPA v18 LambdaRank + Pre-flight Optimization + VR 200q Credibility Run

**CalVer:** 2026.04.55 (sequence increment from 2026.04.54)

**Authored:** 2026-04-22

**Previous Milestone:** 2026.04.54 — "JEPA v17 RankNet Loss + Gemma4 VR Diagnostic + PSV PaCoRe K=2"

---

## Executive Summary

Milestone .54 closed all 7 planned experiments and retired 7 legacy slowest-5 experiments, freeing 322 min/milestone. Despite operational wins, three strategic blockers persist:

1. **JEPA anti-correlation** (RETRO-CRITICAL): v15 AUC=0.4751, v16=0.4759, v17=0.4819 — three consecutive milestones below random 0.5. Root cause identified: pairwise RankNet allows intra-pair hedging when pairs are drawn from a shallow pool. Fix: LambdaRank listwise loss on FoVer v2 data (1000+ pairs, 5+ steps/question).
2. **Verifiable Reasoning never positive** (RETRO-033): 19 consecutive attempts, signed_improvement always ≤ 0. Root cause hypothesis: 100q is insufficient signal; Gemma4 may have constraint-induced distortion (arXiv 2601.01490). Fix: 200q scale + graduated threshold search.
3. **Pre-flight overhead at 14.6% wall time** (562 min, 14 consecutive milestones without incremental selection): Now the largest recoverable overhead, exceeds retired slowest-5 combined (322 min). Fix: pytest-changed incremental test selection.

Milestone .55 designs 13 experiments across 5 phases addressing all three blockers plus two new research tracks seeded from 2026 arxiv scan.

---

## Three Biggest Gaps (PRD vs. Current State)

### Gap 1: JEPA Predictive Verification Blocked (FR-11, FR-12)

The PRD requires Tier 2 JEPA as a working pre-generative constraint violation predictor. After 3 consecutive milestones of below-random performance, this is the most critical blocker to Phase 1 completion. FoVer v2 (1000+ labeled step-level pairs from Exp 712) now provides the listwise training signal needed. LambdaRank ranks ALL steps per question jointly, preventing intra-pair hedging that plagued v17's pairwise RankNet approach.

**New papers enabling the fix:** arXiv 2512.19171 (JEPA-Reasoner pre-generative latent verification), arXiv 2504.10559 (ActPRM active uncertainty weighting for step-level labels).

### Gap 2: Verifiable Reasoning Has No Positive Signal (FR-12)

19 consecutive VR attempts with signed_improvement ≤ 0 means FR-12 Tier 3 verification provides no measurable improvement to answer quality. The constraint-induced distortion paper (arXiv 2601.01490) explains why smaller models (Gemma4-A4B) may degrade: constraint satisfaction pressure shifts internal representations away from factual recall, producing distorted outputs when thresholds are tight. Graduated threshold search + 200q scale are the next diagnostic steps.

### Gap 3: Autonomous Self-Learning Loop Not Deployed (FR-11)

FR-11 requires an online weight-update loop: JEPA detects constraint violations → updates model weights → improves future generations. The FR-11 relay experiment (Exp 719) is gated on JEPA v18 working (AUC ≥ 0.75). Until JEPA is fixed, FR-11 cannot be wired up. This gap closes only if Gap 1 closes first.

---

## Dependency Graph

```
Exp 716 (Pre-flight v7)           — no dependencies
Exp 717 (JEPA v18 LambdaRank)     — no dependencies (uses FoVer v2 from Exp 712)
Exp 718 (JEPA v18 Cascade Deploy) — GATED on Exp 717 AUC >= 0.75
Exp 719 (FR-11 Relay)             — GATED on Exp 718
Exp 720 (VR 200q Qwen3.5-0.8B)   — no dependencies
Exp 721 (VR Gemma4 Graduated)     — no dependencies
Exp 722 (PSV Pool Exhaustion)     — no dependencies
Exp 723 (PSV PaCoRe v2)           — GATED on Exp 722 diagnosis
Exp 724 (KAN Distillation v3)     — no dependencies
Exp 725 (SC-Energy v2 FoVer v2)   — no dependencies
Exp 726 (JEPAReasonerProbe)       — no dependencies
Exp 727 (VarGranularity Gate)     — no dependencies
Exp 728 (Operational Retro)       — all other experiments complete
```

---

## Phase 0: Operational Pre-flight

### Exp 716: Pre-flight v7 + Incremental Test Selection

**Goal:** Reduce pre-flight overhead from 562 min (14.6% wall time) below 200 min by implementing incremental test selection — only run tests for modules changed since last passing run.

**Motivation:** Pre-flight is now the largest recoverable overhead. The slowest-5 governance (Exp 703) freed 322 min; pre-flight incremental selection targets a comparable 362 min reduction. Also monitor Exp 527 (52 min, new slowest-5 first appearance — Live 100q Precision v8) for BatchedInferenceRunner application.

**Implementation:**
- Integrate `pytest-changed` or equivalent git-diff-based test selection
- Track `git diff --name-only HEAD~1` → map changed Python/Rust modules → select only dependent test files
- Fallback: run full suite if diff exceeds 20 files or touches core EBM modules
- Apply BatchedInferenceRunner to Exp 527 if it remains in slowest-5

**Success criteria:**
- Pre-flight wall time < 200 min (from 562 min)
- Zero test regressions (full suite still passes on weekly cadence)
- Exp 527 no longer in slowest-5

**Deliverable:** `results/experiment_716_preflight_v7.json`

**Hardware:** CPU only

---

## Phase 1: JEPA v18 LambdaRank (RETRO-CRITICAL)

### Exp 717: JEPA v18 LambdaRank Listwise Loss on FoVer v2

**Goal:** Train JEPA v18 using listwise LambdaRank loss on the FoVer v2 dataset (1000+ labeled step-level pairs, 5+ steps/question), targeting OOD AUC ≥ 0.75 on GSM8K questions 500-699.

**Root cause of v17 failure:** Pairwise RankNet compares two steps at a time and allows the model to hedge — if a pair is drawn from similar-quality steps, the loss gradient is near zero and the model learns nothing discriminative. With only 10 steps/question in training data, many pairs have similar quality.

**LambdaRank fix:** Rank ALL steps per question jointly using an NDCG surrogate. The loss is computed over the full ranked list, preventing any single pair from dominating. FoVer v2 provides 5+ steps/question (median), enabling non-trivial listwise ranking.

**Implementation based on arXiv 2512.19171 (JEPA-Reasoner):** Extract question-end hidden states from JEPA encoder, apply listwise ranking head. ActPRM uncertainty weighting (arXiv 2504.10559): weight training examples by disagreement between Z3 and PDDL labels — high disagreement = uncertain = down-weight.

**Architecture:**
```
FoVer v2 pairs (1000+, 5+ steps/q)
  → Group by question
  → LambdaRank listwise loss (NDCG surrogate)
  → ActPRM uncertainty weighting
  → JEPA encoder + ranking head
  → OOD eval: GSM8K 500-699
```

**Success criteria:**
- OOD AUC ≥ 0.75 on GSM8K 500-699 (from v17 AUC=0.4819)
- Training converges (val loss decreasing at epoch 10)
- If AUC ≥ 0.75: proceed to Exp 718
- If AUC < 0.75: file RETRO-CRITICAL update, report root cause, do NOT proceed to Exp 718

**Deliverable:** `results/experiment_717_jepa_v18_lambdarank.json`

**Hardware:** GPU required (RTX 3090)

---

### Exp 718: JEPA v18 Cascade Deploy (GATED on Exp 717)

**Goal:** Deploy JEPA v18 into the verification cascade as Tier 2, replacing v17. Wire up to the existing cascade router.

**Gate condition:** Only runs if Exp 717 AUC ≥ 0.75. If gate fails, this experiment is skipped and logged as GATED-BLOCKED.

**Implementation:**
- Load JEPA v18 checkpoint from Exp 717
- Update `python/carnot/cascade/tier2_jepa.py` to load v18 weights
- Run cascade integration smoke test on 50q GSM8K validation set
- Measure cascade latency delta (must be < +5ms per query)

**Success criteria:**
- Cascade AUC ≥ 0.70 on 50q validation set
- Latency delta < +5ms
- All cascade integration tests pass

**Deliverable:** `results/experiment_718_jepa_v18_cascade.json`

**Hardware:** GPU required (RTX 3090)

---

### Exp 719: FR-11 Tier 2 Relay (GATED on Exp 718, FR-11 MANDATORY)

**Goal:** Wire FR-11 autonomous self-learning loop at Tier 2: JEPA detects constraint violation → fires weight-update signal → Tier 1 online updater adjusts model weights.

**Gate condition:** Only runs if Exp 718 cascade deploy succeeds. If gate fails, GATED-BLOCKED.

**This is a MANDATORY FR-11 experiment.** The PRD requires an online learning loop and this is the first wiring attempt.

**Implementation:**
- JEPA violation signal → event bus → Tier 1 weight updater callback
- Throttle: maximum 1 update per 10 queries to prevent oscillation
- Log all update events to `ops/fr11_relay_log.jsonl`
- Measure FR-11 feedback loop latency (signal → update ack < 200ms)

**Success criteria:**
- At least 1 successful relay event logged in 50q test run
- FR-11 loop latency < 200ms
- No cascade regressions (AUC delta < -0.02)

**Deliverable:** `results/experiment_719_fr11_relay.json`

**Hardware:** GPU required (RTX 3090)

---

## Phase 2: VR Scale Validation (RETRO-033)

### Exp 720: Live VR Attempt #20 — 200q Qwen3.5-0.8B

**Goal:** Run Verifiable Reasoning at 200q scale with Qwen3.5-0.8B to determine whether signed_improvement > 0 is achievable at larger scale.

**Motivation (RETRO-033):** 19 consecutive VR attempts all returned signed_improvement ≤ 0 at 100q scale. The signal may be too noisy at 100q. Doubling to 200q either provides the clean signal needed or confirms that VR itself is the wrong approach for this model family.

**Implementation:**
- Use BatchedInferenceRunner with 8q batches (timeout = 8*60s per batch)
- Use LongRunBenchmarkExecutor for 200q total
- Model: Qwen3.5-0.8B (GPU 0, RTX 3090)
- Measure signed_improvement at 50q, 100q, 150q, 200q checkpoints

**Success criteria:**
- signed_improvement > 0 at 200q (first positive VR result)
- If negative at 200q: file RETRO-033 resolution — VR not viable for this model/scale, pivot to abstain mode

**Deliverable:** `results/experiment_720_vr_200q_qwen.json`

**Hardware:** GPU required (RTX 3090)

---

### Exp 721: Live VR Gemma4 Graduated Threshold Search

**Goal:** Test VR on Gemma4-A4B with graduated constraint thresholds [0.10, 0.20, 0.30, 0.40] and abstain mode to diagnose constraint-induced distortion (arXiv 2601.01490).

**Motivation:** arXiv 2601.01490 shows that tight constraint satisfaction pressure shifts internal representations away from factual recall in smaller models — producing distorted, hallucinated outputs that score worse than unconstrained generation. Gemma4-A4B (4B active params) is in the size range where this distortion occurs.

**Hypothesis:** A looser threshold (0.30-0.40) or abstain mode (only apply constraint when confidence > 0.9) may avoid distortion while still catching high-confidence violations.

**Implementation:**
- Test 5 conditions: threshold ∈ {0.10, 0.20, 0.30, 0.40, abstain}
- 50q per condition = 250q total
- Model: Gemma4-A4B-GGUF (GPU 1, RTX 3090)
- Measure signed_improvement per condition

**Success criteria:**
- At least one threshold condition produces signed_improvement > 0
- Identify optimal threshold for Gemma4 deployment

**Deliverable:** `results/experiment_721_vr_gemma4_graduated.json`

**Hardware:** GPU required (RTX 3090)

---

## Phase 3: PSV Self-Play Recovery

### Exp 722: PSV Root Cause Analysis — Question Pool Exhaustion Diagnosis

**Goal:** Diagnose whether PSV degradation (slope=0.001212 positive in .54, up from 0.004242 in .53, improvement in rate but still degrading) is caused by question pool exhaustion — model memorizes fixed 10-question pool after 10 iterations.

**Hypothesis:** With a fixed 10-question pool, after ~10 PSV iterations, the model has seen each question multiple times and its constraint-satisfaction updates overfit to those specific questions. When evaluated on held-out questions, performance has deteriorated.

**Controlled experiment:**
- Condition A: Fixed 10-question pool (current, reproduces degradation)
- Condition B: Rotating 100-question pool, sample 10 random per iteration
- Run both for 20 iterations, measure FP rate trend slope

**Success criteria:**
- If Condition B slope < 0 (improving): confirmed pool exhaustion, proceed to Exp 723
- If both conditions degrade: pool exhaustion not the root cause, file new RETRO

**Deliverable:** `results/experiment_722_psv_pool_exhaustion.json`

**Hardware:** CPU (diagnostic, no model inference needed — analyze existing PSV logs)

---

### Exp 723: PSV PaCoRe v2 — Diverse Pool + K=2 (GATED on Exp 722)

**Goal:** Retrain PSV with rotating 100-question pool, K=2 parallel chains, targeting fp_rate_trend_slope < 0 (improving).

**Gate condition:** Only runs if Exp 722 confirms pool exhaustion as root cause.

**Implementation:**
- Build 100-question diverse pool from GSM8K + MATH + ARC-Challenge
- Sample 10 random questions per PSV iteration
- K=2 PaCoRe chains with diversity penalty (reward questions where chains disagree)
- Run 30 iterations

**Success criteria:**
- fp_rate_trend_slope < 0 (improving, from current +0.001212)
- FP rate at iteration 30 < FP rate at iteration 0

**Deliverable:** `results/experiment_723_psv_pcore_v2.json`

**Hardware:** GPU required (RTX 3090)

---

## Phase 4: New Research Tracks

### Exp 724: KAN Distillation v3 — 3000 Examples + 16 Knots

**Goal:** Push KAN distillation toward the 0.90 AUROC gate by expanding training data to 3000 examples and increasing knot count from 8 to 16.

**Current state:** v2 AUROC=0.8747, trend toward 0.90. More data and more expressive KAN should close the gap.

**Implementation:**
- Generate 3000 labeled prompt-injection examples (1500 positive, 1500 negative)
- KAN with 16 knots per activation (from v2's 8 knots)
- Same training procedure as v2 (Adam, lr=1e-3, 100 epochs)

**Success criteria:**
- AUROC ≥ 0.90 on held-out test set (gate for KAN deployment)
- If AUROC ≥ 0.90: deploy to Tier 0b, file capability update

**Deliverable:** `results/experiment_724_kan_distill_v3.json`

**Hardware:** CPU only (KAN training is fast)

---

### Exp 725: SC-Energy v2 on FoVer v2

**Goal:** Retrain SC-Energy (self-consistency energy, Tier 2.9) on FoVer v2 labeled pairs, targeting AUC ≥ 0.75.

**Motivation:** FoVer v2 provides 1000+ step-level pairs with both Z3 and PDDL labels — richer supervision than the original SC-Energy training data. The combination of Z3 (symbolic) and PDDL (planning) labels gives SC-Energy a stronger consistency signal.

**Implementation:**
- Load FoVer v2 pairs from Exp 712 output
- Retrain SC-Energy EBM with FoVer v2 labels as supervision
- Evaluate on OOD GSM8K 500-699

**Success criteria:**
- OOD AUC ≥ 0.75 (from current SC-Energy v1 baseline)
- Tier 2.9 cascade integration passes smoke test

**Deliverable:** `results/experiment_725_sc_energy_v2.json`

**Hardware:** CPU (EBM training is fast)

---

### Exp 726: JEPAReasonerProbe — Pre-generative Latent Verification

**Goal:** Implement the JEPA-Reasoner approach from arXiv 2512.19171: extract question-end hidden states BEFORE generation begins, train a lightweight probe (2-layer MLP) on FoVer v2 pairs to predict constraint violations from latent representations.

**Motivation:** arXiv 2512.19171 shows that LLM hidden states at question-end contain strong signal about whether the coming generation will satisfy constraints — even before generation starts. This enables <1ms latency verification (just a probe forward pass) vs. the current JEPA approach which requires full step generation.

**Implementation:**
- Model: Qwen3.5-0.8B (GPU, extract hidden states layer 16)
- Extract hidden states at question-end for all FoVer v2 examples
- Train 2-layer MLP probe: hidden_dim → 256 → 1, BCELoss
- Evaluate: AUC on OOD GSM8K 500-699, latency measurement

**Success criteria:**
- OOD AUC ≥ 0.75 (same gate as JEPA)
- Inference latency < 1ms (probe-only, not counting LLM forward pass)
- If both met: propose as Tier 2.1 (latency-optimized JEPA alternative)

**Deliverable:** `results/experiment_726_jepa_reasoner_probe.json`

**Hardware:** GPU required (RTX 3090, hidden state extraction)

---

### Exp 727: Variable Granularity Verification Gate

**Goal:** Implement adaptive compute allocation from arXiv 2505.11730: use EORM confidence gate to decide whether to run expensive Tier 3 Ising sampler, targeting >50% Ising skip rate with <5% false-negative delta.

**Motivation:** arXiv 2505.11730 shows that applying verification at variable granularity based on confidence — skip expensive verification when cheap tiers are highly confident — can achieve near-identical accuracy at 40-60% lower compute cost. Tier 3 Ising is the most expensive tier.

**Implementation:**
- EORM confidence score from existing Tier 0h
- Gate: if EORM confidence > 0.92, skip Tier 3 Ising (mark as verified-fast)
- If EORM confidence < 0.92, run Tier 3 Ising as normal
- Evaluate on 200q test set: Ising skip rate, FN delta vs. full cascade

**Success criteria:**
- Ising skip rate > 50% (saves >50% Tier 3 compute)
- False-negative delta < 5% vs. full cascade (minimal accuracy loss)
- Latency reduction > 30% on 200q benchmark

**Deliverable:** `results/experiment_727_vargran_gate.json`

**Hardware:** CPU (Ising already implemented, gate logic is CPU-only)

---

## Phase 5: Retrospective

### Exp 728: Milestone 2026.04.55 Operational Retrospective

**Goal:** Measure wall time, slowest-5, and strategic outcomes vs. .55 plan. Feed operational improvements back into process.

**Template:** Follow exact format from `results/operational_retro_2026_04_54.json`.

**Key questions to answer:**
1. Did JEPA v18 achieve AUC ≥ 0.75? (RETRO-CRITICAL resolution)
2. Did VR ever produce signed_improvement > 0? (RETRO-033 resolution)
3. Did pre-flight overhead fall below 200 min?
4. Is PSV now improving (slope < 0)?
5. What are the new slowest-5?

**Deliverable:** `results/operational_retro_2026_04_55.json`

**Hardware:** CPU only

---

## Experiment Summary Table

| Exp | Title | Phase | Hardware | Gate | Deliverable |
|-----|-------|-------|----------|------|-------------|
| 716 | Pre-flight v7 + Incremental Test Selection | 0 | CPU | None | experiment_716_preflight_v7.json |
| 717 | JEPA v18 LambdaRank | 1 | GPU | None | experiment_717_jepa_v18_lambdarank.json |
| 718 | JEPA v18 Cascade Deploy | 1 | GPU | Exp 717 AUC≥0.75 | experiment_718_jepa_v18_cascade.json |
| 719 | FR-11 Tier 2 Relay | 1 | GPU | Exp 718 pass | experiment_719_fr11_relay.json |
| 720 | Live VR 200q Qwen3.5-0.8B | 2 | GPU | None | experiment_720_vr_200q_qwen.json |
| 721 | VR Gemma4 Graduated Threshold | 2 | GPU | None | experiment_721_vr_gemma4_graduated.json |
| 722 | PSV Pool Exhaustion Diagnosis | 3 | CPU | None | experiment_722_psv_pool_exhaustion.json |
| 723 | PSV PaCoRe v2 Diverse Pool | 3 | GPU | Exp 722 confirm | experiment_723_psv_pcore_v2.json |
| 724 | KAN Distillation v3 | 4 | CPU | None | experiment_724_kan_distill_v3.json |
| 725 | SC-Energy v2 FoVer v2 | 4 | CPU | None | experiment_725_sc_energy_v2.json |
| 726 | JEPAReasonerProbe | 4 | GPU | None | experiment_726_jepa_reasoner_probe.json |
| 727 | Variable Granularity Gate | 4 | CPU | None | experiment_727_vargran_gate.json |
| 728 | Operational Retrospective | 5 | CPU | All complete | operational_retro_2026_04_55.json |

---

## New Papers Filed (2026-04-22 Scan)

| Paper | ArXiv ID | Experiment |
|-------|----------|------------|
| JEPA-Reasoner: Pre-generative Latent Verification | 2512.19171 | Exp 726 |
| Variable Granularity Search for Adaptive Verification | 2505.11730 | Exp 727 |
| ActPRM: Active Learning for Process Reward Models | 2504.10559 | Exp 717 (weighting) |
| Constraint-Induced Distortion in Small LLMs | 2601.01490 | Exp 721 (diagnosis) |

---

## Hardware Requirements

| Experiment | GPU | Model | VRAM |
|------------|-----|-------|------|
| 717 | RTX 3090 GPU 0 | JEPA encoder (Qwen3.5-0.8B base) | ~8 GB |
| 718 | RTX 3090 GPU 0 | JEPA v18 + cascade | ~8 GB |
| 719 | RTX 3090 GPU 0 | Full cascade | ~12 GB |
| 720 | RTX 3090 GPU 0 | Qwen3.5-0.8B | ~6 GB |
| 721 | RTX 3090 GPU 1 | Gemma4-A4B-GGUF | ~8 GB |
| 723 | RTX 3090 GPU 0 | PSV model (Qwen3.5-0.8B) | ~6 GB |
| 726 | RTX 3090 GPU 0 | Qwen3.5-0.8B (hidden states) | ~8 GB |

GPU 0 and GPU 1 can run independently on dual RTX 3090 setup.

---

## Success Criteria Summary

| Blocker | Target | Experiment | Resolution |
|---------|--------|------------|------------|
| JEPA anti-correlation (RETRO-CRITICAL) | OOD AUC ≥ 0.75 | Exp 717 | LambdaRank listwise on FoVer v2 |
| VR never positive (RETRO-033) | signed_improvement > 0 | Exp 720/721 | 200q scale + graduated threshold |
| Pre-flight overhead 14.6% | Wall time < 200 min | Exp 716 | pytest-changed incremental selection |
| PSV degrading | fp_rate_trend_slope < 0 | Exp 722/723 | Diverse 100-q pool |
| FR-11 not wired | Relay event logged | Exp 719 | Gated on JEPA v18 success |
| KAN not at 0.90 gate | AUROC ≥ 0.90 | Exp 724 | 3000 examples + 16 knots |

---

## Risk Register

| Risk | Probability | Mitigation |
|------|-------------|------------|
| JEPA v18 still below 0.5 | Medium | If 4th consecutive failure, pivot to JEPAReasonerProbe (Exp 726) as alternative Tier 2 |
| VR 200q still negative | Medium | Close RETRO-033 as "VR not viable at current model scale", remove from roadmap |
| PSV pool exhaustion not confirmed | Low | Alternative hypothesis: model capacity too low; test with Qwen3.5-7B |
| Exp 527 grows beyond 60 min | Low | Apply BatchedInferenceRunner immediately in Exp 716 |
| KV260 Vivado still unavailable | High | Defer all FPGA experiments to next milestone (Vivado install blocked) |

---

## Conductor Execution Notes

- Run Exp 716 first (frees wall time for all subsequent experiments).
- Exps 717-719 must run sequentially (each gated on previous).
- Exps 720, 721, 722, 724, 725, 726, 727 can run in any order (no mutual dependencies).
- Exp 723 must follow Exp 722.
- Exp 728 must be last.
- GPU experiments: prefer GPU 0 for Qwen/JEPA work, GPU 1 for Gemma4.

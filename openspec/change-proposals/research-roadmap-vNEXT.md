# Carnot Research Roadmap v29: JEPA Real Training, Z3 Formal Extraction, KV260 FPGA, and Credible Full-Scale Benchmarks

**Created:** 2026-04-14
**Milestone:** 2026.04.29
**Status:** Planned (activates when milestone 2026.04.22 completes)
**Supersedes:** Milestone 2026.04.22 — "Apple Adversarial Completion, Dual-Energy Gate, Constraint Addition, and NPU Hardware"
**Informed by:** Exps 294-306, operational retrospective 2026.04.22, v28 carry-forwards
**External inputs (new in v29):**
- LLM-JEPA (2509.14252) — JEPA applied to LLMs, outperforms standard training objectives
- Emergent Formal Verification (2603.21149) — Z3 SMT across 6 domains, 100% accuracy, 0% FP
- Correctness-Guaranteed Code Gen via Constrained Decoding (2508.15866) — dynamic parser tree for guided decoding
- Hybrid FPGA Ising Decomposition with COBI (2602.15985) — 77.5μs convergence, <10mW, graph decomposition
- RL-Tuned LLMs as EBMs (2512.18730) — optimal RL policy = EBM over base distribution
- Z3 Security Vulnerability Verification (2604.05292) — 3,500 LLM artifacts, 55.8% vulnerability rate
- Energy Matching (2504.10612) — unifies flow matching + EBMs, eliminates MCMC sampling bottleneck

---

## What 2026.04.22 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| GPU stall diagnosis + Apple baseline | 294 | **COMPLETE** | pre_warm_fix confirmed; logits produced |
| Apple adversarial verify-repair | 295 | **COMPLETE** | 12-cell benchmark with pre-warm |
| Apple adversarial analysis | 296 | **COMPLETE** | Result classified |
| SemanticEnergyExtractor + VarEntropyProbe | 297 | **IMPLEMENTED** | 3-signal extraction-free detection system |
| PrefillUncertaintyProbe | 298 | **IMPLEMENTED** | Pre-generation hallucination gate live |
| JEPA retrain on synthetic | 299 | **TARGETS_MET (synthetic)** | TP=1.0/FP=0.0 but training_source=synthetic_fallback |
| ConstraintGenerator from CaseMemory | 300 | **IMPLEMENTED** | Tier 2→Tier 1 constraint addition with soundness bounds |
| Confidence-weighted repair gating | 301 | **IMPLEMENTED** | REQ-VERIFY-081/082, gates repair on confidence≥0.8 |
| Integrated self-learning benchmark | 302 | **COMPLETE** | Tier 1+2 integrated; improvement_delta measured |
| AMD XDNA NPU unblock | 303 | **BLOCKED** | blocked_prereq: ninja+openblas still missing |
| HuggingFace FCV upload | 304 | **FCV LIVE** | carnot-formal-claim-verifier-v1 published |
| ExperimentTemplate + BatchedInferenceRunner | 306 | **COMPLETE** | <0.5s overhead; 3975 tests pass |

**Milestone-level conclusion:**
2026.04.22 fixed the GPU stall, completed the Apple adversarial benchmark, implemented the 3-signal
extraction-free detection system, and deployed Tier 1+2 self-learning infrastructure. The critical
remaining gaps are: (1) JEPA trained only on synthetic data — real logits now exist from Exps
294/295 and must be used; (2) Z3 formal extraction pipeline is not yet implemented — constraint
extraction remains the #1 accuracy bottleneck; (3) no full-scale benchmark with confidence
intervals exists — all results are on small cohorts (50-200 questions).

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Tier 3 JEPA is synthetic-only — no real training data path

The JEPA predictive verification architecture is correct (Exp 291: TP=1.0/FP=0.0), but it was
trained on synthetic data. Real logits from the Apple adversarial benchmark now exist:
`data/research/logits_294_*.npy` and `data/research/logits_295_*.npy` (produced by Exps 294/295).

The training objective from LLM-JEPA (2509.14252) provides the right architecture: encode the
partial response (first N tokens), predict the embedding of the final response, measure energy
between prediction and actual. High energy = violation likely. The (partial_response,
final_violation) pairs are extractable from Apple adversarial logit files right now.

Without real training, the JEPA fast-path gate cannot reliably route queries — it would skip
expensive Ising verification for queries that actually need it. This is the key Tier 3 gap.

**This milestone's priority #1: Train JEPA on real Apple adversarial logits, then measure
whether the fast-path gate reduces pipeline latency without hurting accuracy.**

### Gap 2: Constraint extraction is still regex-based for IT models

The `ArithmeticExtractor` detected 0 violations on Gemma4-E4B-it (Exp 203). The LLM extractor
(Exp 207) has better FP rate (1/91) but still misses real errors (0/9 wrong answers detected).
The FormalClaimVerifier (Exp 293/304) works for arithmetic/comparison claims but requires
manually formatted claims.

Emergent Formal Verification (2603.21149) demonstrates that using an LLM to translate chain-of-
thought responses into Z3 SMT assertions achieves 100% accuracy with 0% FP across 6 domains.
Z3 Security Verification (2604.05292) extends this to code, finding 55.8% vulnerability rate
across 3,500 LLM artifacts.

The missing piece: an `NL2Z3Extractor` that takes a chain-of-thought response and uses a second
LLM call to generate Z3 Python code that verifies each reasoning step. Unlike regex (catches only
`a+b=c` patterns), Z3 can verify logical entailment, range constraints, type invariants, and
semantic consistency.

**This milestone's priority #2: Implement NL2Z3Extractor and benchmark it against ArithmeticExtractor
and LLMExtractor on the same IT-model corpus. Goal: FP < 5%, TP > 25%.**

### Gap 3: No published full-scale benchmark with confidence intervals

Carnot's verify-repair results are all on small cohorts (20-200 questions). No confidence intervals.
No comparison to published baselines. The research program explicitly requires:
- 200+ GSM8K + 50 HumanEval on LIVE GPU with inference_mode="live_gpu"
- Error bars at 95% CI using bootstrap or Wilson interval
- Comparison to published Qwen3.5-0.8B and Gemma4-E4B-it baselines

Correctness-Guaranteed Code Gen (2508.15866) provides a benchmark comparison point for HumanEval.
The Apple adversarial results (Exp 295) can be extended to full-corpus analysis.

**This milestone's priority #3: Run full 400-question Apple adversarial + 200-question standard
GSM8K + 50 HumanEval with bootstrap confidence intervals. Report inference_mode="live_gpu".**

---

## Architecture: v29 Additions

```
[Input Query]
    │
    ├─[PrefillUncertaintyProbe]─── prefill uncertainty (EXISTING, Exp 298)
    │      arXiv 2603.19562       ↓
    │                          [SKIP if low uncertainty — fast path]
    │
    ▼
[LLM Generation]
    │
    ├─[JEPA Fast-Path Gate]──────── violation predicted? (NEW: trained on real logits)
    │  arXiv 2509.14252            ↓ if "high energy likely"
    │                          [TRIGGER full Ising verification]
    │                          [SKIP Ising if "low energy" — 10x faster]
    │
    ├─[SpilledEnergyExtractor]─── spilled energy  (EXISTING, Exp 285)
    │
    ├─[SemanticEnergyExtractor]── confident-wrong  (EXISTING, Exp 297)
    │
    ├─[VarEntropyProbe]──────────── entropy variance (EXISTING, Exp 297)
    │
    └─[NL2Z3Extractor]──────────── Z3 SMT assertions  (NEW: this milestone)
           arXiv 2603.21149        ↓
                               [Z3 SAT/UNSAT check]
                               └─[UNSAT → trigger Ising repair]
                               └─[SAT + low confidence → Ising verify anyway]
    │
    ▼
[Ising Verification] ← routed from JEPA gate or high-confidence extractors
    │
    ├─[ConfidenceVerifier]─────── energy → confidence ≥ 0.8 → repair (EXISTING)
    │      arXiv 2602.03979
    │
    └─[Repair if needed]
    │
    ▼
[Tier 1: Online constraint weights updated] ← per-constraint precision (EXISTING)
[Tier 2: CaseMemory → ConstraintGenerator] ← constraint addition (EXISTING, Exp 300)
[Tier 3: JEPA trained on accumulated logs] ← (NEW: this milestone, real logits)
```

**New hardware path (KV260 FPGA — this milestone):**
```
[Ising Verification]
    │
    └─[FpgaBackend.sample()]
           ↓
    [KV260 PYNQ overlay] ← if CARNOT_KV260_BITFILE set (EXISTING design, Exp 289)
           ↓
    [AXI-Lite → spin states] ← 77.5μs target (arXiv 2602.15985)
           ↓
    [CPU fallback] ← if overlay not loaded (always works)
```

---

## Phase Descriptions

### Phase 1: JEPA Real Training + Tier 3 Integration (Exps 307-309)

Train the JEPA violation predictor on real Apple adversarial logits from Exps 294/295. Wire it as
a live fast-path gate in `VerifyRepairPipeline`. Measure whether the gate reduces latency while
maintaining accuracy.

**Deliverables:**
- `results/experiment_307_jepa_real_training.json` — training metrics on real logits
- `results/jepa_predictor_307.onnx` — updated JEPA model trained on real data
- `python/carnot/pipeline/jepa_fast_path.py` — live gate integration (additive to pipeline)
- `results/experiment_308_jepa_gate_benchmark.json` — latency with/without gate
- `results/experiment_309_tier3_pipeline.json` — Tier 3 end-to-end results

**Success criteria:**
- JEPA trained on ≥500 (partial, violation) pairs from real logit files
- Fast-path skip rate ≥ 30% at TP ≥ 0.85 on held-out set (or honest report if not met)
- Pipeline latency reduction ≥ 15% on 50-question batch when gate enabled (or honest report)

### Phase 2: Z3 Formal Extraction (Exps 310-312)

Implement `NL2Z3Extractor` using a second LLM call to translate chain-of-thought into Z3 Python
assertions. Benchmark against existing extractors on the same IT-model corpus.

**Deliverables:**
- `python/carnot/pipeline/nl2z3_extractor.py` — NL→Z3 extractor
- `results/experiment_310_nl2z3_results.json` — extraction results on 50 IT-model responses
- `results/experiment_311_extractor_benchmark.json` — FP/TP comparison: regex vs LLM vs Z3
- `python/carnot/pipeline/z3_gated_repair.py` — Z3 UNSAT → repair gate

**Success criteria:**
- FP rate ≤ 5% on correct IT-model responses (vs regex 3/20=15% false positives)
- TP rate > 0% on wrong IT-model responses (any improvement over current 0/9=0%)
- Z3 verification completes in < 2s per response

### Phase 3: KV260 FPGA Bring-up + NPU Retry (Exps 313-314)

Bring up the KV260 FPGA board. Test PYNQ overlay loading and AXI-Lite Ising sampling using the
design from Exp 228. Retry AMD XDNA NPU after installing missing prerequisites.

**Deliverables:**
- `results/experiment_313_kv260_bringup.json` — PYNQ overlay status, spin sampling result
- `results/experiment_314_npu_prereq_install.json` — NPU status post-prereq install

**Success criteria:**
- KV260: overlay loads OR honest blocked artifact with specific blocker
- At least 1 AXI-Lite spin-state round-trip OR honest timeout artifact
- NPU: blocked_prereq resolved OR new specific blocker documented with install steps

### Phase 4: Full-Scale Benchmarks + HuggingFace + Self-Learning + Retro (Exps 315-319)

Phased full-scale benchmark per lessons-learned (script first, execute second). HuggingFace
publish. End-to-end Tier 1+2+3 self-learning relay. Operational retrospective.

**Deliverables:**
- `scripts/experiment_315_fullscale_script.py` — benchmark script (deliverable = script only)
- `results/experiment_316_fullscale_results.json` — execution with 95% CI
- `results/experiment_317_hf_publish.json` — updated HF model READMEs + any new models
- `results/experiment_318_self_learning_relay.json` — Tiers 1+2+3 relay
- `results/operational_retro_2026_04_29.json` — retrospective

---

## Dependency Graph

```
Exp 307 → Exp 308 → Exp 309 ──────────────────────────────────┐
                                                                │
Exp 310 → Exp 311 → Exp 312 ──────────────────────────────────┤
                                                                ▼
Exp 313 (parallel, hardware)                          Exp 315 → Exp 316
Exp 314 (parallel, hardware)                                    │
                                                       Exp 317 (parallel)
                                                       Exp 318 → after Exp 309+312
                                                                │
                                                       Exp 319 (retro, runs last)
```

Phases 1-3 can run in parallel with each other. Exp 318 (self-learning relay) depends on Exps
309 (JEPA gate live) and 312 (Z3 extractor live). Exp 319 (retro) runs last.

---

## Hardware Requirements

| Experiment | Hardware | Required | Notes |
|------------|----------|----------|-------|
| Exps 307-309 | 2x RTX 3090 | YES | JEPA training + real LLM inference |
| Exps 310-312 | CPU only | NO | LLM extraction + Z3 on CPU |
| Exp 313 | KV260 FPGA | DESIRABLE | Emit blocked artifact if not arrived |
| Exp 314 | CPU + sudo | NO | System package install required |
| Exps 315-316 | 2x RTX 3090 | YES | Live LLM inference for 400+ questions |
| Exps 317-319 | CPU | NO | Publishing + retrospective |

**KV260 note:** If the KV260 hasn't arrived by Exp 313, emit a `blocked_shipping` honest artifact.
Do not block the milestone on hardware arrival — continue with Exp 314.

---

## Carry-Forward Items from 2026.04.22

| Item | Status | Action |
|------|--------|--------|
| AMD XDNA NPU (Exp 303) | blocked_prereq | Retry in Exp 314: `sudo pacman -S ninja openblas` |
| carnot-joint-constraint-v1 HF model | skipped (no .safetensors) | Create honest placeholder in Exp 317 |
| RETRO-001 conductor 45-min timeout | not resolved | Add to 2026.04.29 retro action items |
| RETRO-002 GPU monitor in conductor | not resolved | Add to 2026.04.29 retro action items |
| research-roadmap-vNEXT.md | superseded | This document supersedes it |

---

## Papers Added to research-references.md This Milestone

| Paper | ArXiv ID | Priority | Experiment |
|-------|----------|----------|-----------|
| LLM-JEPA | 2509.14252 | HIGH | Exps 307-309 |
| Emergent Formal Verification (Z3) | 2603.21149 | HIGH | Exps 310-312 |
| Correctness-Guaranteed Code Gen | 2508.15866 | HIGH | Exps 315-316 |
| Hybrid FPGA Ising Decomposition | 2602.15985 | MEDIUM | Exp 313 |
| RL-Tuned LLMs as EBMs | 2512.18730 | MEDIUM | Long-term |
| Z3 Security Verification | 2604.05292 | MEDIUM | Exps 310-312 |
| Energy Matching (flow + EBM) | 2504.10612 | LOW | Future training efficiency |

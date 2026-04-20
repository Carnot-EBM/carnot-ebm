# Research Roadmap v43 — Milestone 2026.04.43

**Status:** Proposed
**Milestone:** 2026.04.43
**Title:** Root Cause Surgery — Execution-Based Extraction and PURE JEPA Recovery
**Planned Experiments:** 563–574 (12 experiments)
**Planned Date:** 2026-04-20 onwards

---

## What Milestone 2026.04.42 Proved

Milestone .42 confirmed five important facts and opened three new critical RETROs:

1. **Synthetic barrier partially broken.** Exp 552 collected 50q batch B successfully
   (CARNOT_FORCE_LIVE=1 set correctly in that session). FOVER corpus v2 now contains
   132 pairs with constraint_type_entropy=1.50 bits — enough for model training.
   Exp 549 (exclusion manifest) resolved RETRO-059. Exp 550 wired BatchedInferenceRunner
   into the five slowest legacy scripts.

2. **Extraction TP rate is zero (RETRO-061 CRITICAL).** Exp 554 confirmed that
   VeriCoTStepValidator finds 0 true positives on 25 known-incorrect live responses.
   This propagated downstream: Exps 555 (confidence filtering), 560 (LatentCoT calibrator),
   and 561 (FR-11 relay) all report violation_rate=0.0. The ENTIRE verify-repair pipeline
   is a no-op because extraction never fires. Root cause: VeriCoT/VPRM detect FORMAT
   violations (wrong equation syntax, Z3 UNSAT) but IT models produce correct-FORMAT
   but wrong-ARITHMETIC outputs ("47 + 28 = 76"). The fix is execution-based checking,
   not format checking.

3. **JEPA architecture is wrong (RETRO-060 CRITICAL).** Exp 557 retrained JEPA v9 on
   a 132-pair corpus — significantly larger and more diverse than .41's 24-pair corpus.
   Result: AUC=0.4286, STILL below random. Two consecutive retrains on increasingly
   diverse data produce the same anti-correlated result. The binary BCE loss allows
   the predictor to hedge toward 0.5 everywhere. Fix: replace binary BCE with a
   contrastive margin objective (validated by NUP Probe v4, AUC=1.0 on contrastive loss).

4. **LowRankKAEM needs architectural redesign (RETRO-057).** Exp 559 confirmed that
   a calibration layer alone cannot close the gap — energy_mad_normalized remains
   0.832 vs the 0.05 production tolerance threshold. SVD rank alone cannot recover
   the energy accuracy lost by low-rank projection.

5. **Live 50q A still unrun (RETRO-062).** Exp 551 was blocked because CARNOT_FORCE_LIVE
   was not set when the conductor launched the session. Questions 0-49 are systematically
   absent from the FOVER corpus. Must be run as Exp 563 before any other data work.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: Extraction TP = 0 — Verify-Repair is a No-Op (RETRO-061, FR-12)

**Root cause confirmed:** IT models write arithmetic errors in narrative prose
("we add 47 and 28 to get 76") not in formal syntax. VeriCoT looks for Z3 UNSAT
(logical inconsistency). VPRM looks for pattern violations. Neither fires on a
numerically wrong but syntactically well-formed equation.

**The fix (Caco-style, arXiv 2510.04081):** Parse arithmetic expressions from prose
CoT using a simple AST-compatible extractor, then EXECUTE each expression as Python
and compare the computed result to the stated result. `eval("47+28") = 75 ≠ 76` →
violation detected. No regex patterns. No Z3. This handles any format IT models produce.

This is the #1 critical path item. Every downstream experiment (live verify-repair,
FR-11 relay, JEPA training data) depends on having TP > 0. Do not run another
raw benchmark attempt (RETRO-033 attempt #11) until this is confirmed working.

**Why PRD cares:** FR-12 (Verifiable Reasoning) requires the system to correctly
identify which constraints are violated. Currently it never identifies any violations
on IT-model outputs. This is a complete gap against the core product requirement.

### Gap 2: JEPA Objective is Wrong — Two Retrains, Same Anti-Correlation (RETRO-060)

**Root cause confirmed:** Binary BCE loss (predict correct=1, incorrect=0) allows
the model to reach loss=log(2) by predicting P=0.5 everywhere. The gradient never
forces a useful energy gap. Two consecutive retrains on datasets of 24 and 132 pairs
both converged to AUC ≈ 0.43-0.44 — consistently wrong.

**The fix (PURE, arXiv 2504.15275):** Replace binary BCE with the PURE min-form
PRM objective: score(reasoning_chain) = min(step_score(t) for t in chain). One
bad step forces the whole chain's score low. Training loss = contrastive margin
between correct chains (high min-score) and incorrect chains (low min-score).
This is the same mechanism that made NUP Probe v4 achieve AUC=1.0.

**Why PRD cares:** JEPA (Tier 3 self-learning, FR-11) is meant to predict constraint
violations from partial CoT. A predictor with AUC < 0.5 is worse than random —
it would send the pipeline toward violations rather than away from them.

### Gap 3: Live 50q Batch A Missing — FOVER Corpus Systematically Incomplete (RETRO-062)

**The problem:** GSM8K questions 0-49 were never collected because CARNOT_FORCE_LIVE
was not set at session start for Exp 551. The FOVER corpus v2 (132 pairs) excludes
all questions from that batch. The corpus is large enough for training now (entropy=1.50),
but the missing 50 questions represent a systematic gap in the diversity coverage.

**The fix:** Run Live 50q A as the FIRST experiment of .43, before any other GPU work,
with an explicit pre-flight check that CARNOT_FORCE_LIVE is set. This is Exp 563.

---

## Architecture Diagram (Verification Cascade — .43 State)

```
LLM Response (IT model: Gemma4-E4B-it, Qwen3.5-0.8B)
    │
    ▼
[Tier 0a] CarnotThinkProbe (3-step CoT verdict, ThinkPRM arXiv 2504.16828)
    │  verdict='incorrect' → fast-path violation
    ▼
[Tier 0b] SpilledEnergyDetector (per-token logit discrepancy, arXiv 2602.18671)
    │  high_spill_fraction → pass through
    ▼
[Tier 0c] NUP Probe v4 (contrastive bigram energy, AUC=1.0, Exp 523)
    │  score <= threshold → skip downstream
    ▼
[Tier 0d] HallucinationBasinDetector (latent basin depth, arXiv 2604.04743)
    │  basin_risk <= threshold → skip downstream
    ▼
[Tier 0e] HalluField (thermodynamic energy-path, arXiv 2509.10753) [NEW .43]
    │  low partition variance → skip downstream
    ▼
[Tier 1]  SinkProbe (attention sink concentration, arXiv 2604.10697)
    │  mean_sink >= threshold → skip downstream
    ▼
[Tier 2]  EORM (energy reward model, 55M params, GRPO-retrained on real data)
    │  energy < threshold → verified
    ▼
[Tier 3]  CoACEExtractor [NEW .43] + VerifyRepairPipeline
    │      (Code-Assisted Constraint Extraction, arXiv 2510.04081)
    │      parse arithmetic → execute → compare → violation or verified
    ▼
    Verified / Violated + Repair Suggestion

[Tier 3 alt] PRA EBM Beam Search [NEW .43, arXiv 2604.09482]
    │  K=3 candidates per step, EORM selects minimum energy
    └→ proactive constraint satisfaction during generation
```

**JEPA v10 (Tier 3 fast-path, FR-11):**
```
Partial CoT (first 50%) → JEPA v10 (PURE min-form loss) → energy_score
                                                         → high: run full Tier 3
                                                         → low: skip Tier 3
```

---

## Phase Descriptions

### Phase 0: Data Gap Closure (Exp 563, GPU required)

Close RETRO-062. Run Live 50q A (questions 0-49) with an explicit pre-flight check
that CARNOT_FORCE_LIVE=1 is set before loading any model. This is a simple replication
of Exp 552's successful pattern with a different question slice.

**Success criterion:** n_pairs_collected >= 40 (some latency/OOM headroom).

### Phase 1: Extraction Redesign — RETRO-061 Critical Path (Exps 564-565)

**Exp 564 (CPU-only):** Implement CoACEExtractor (Code-Assisted Constraint Extraction).
The extractor:
1. Parses arithmetic expressions from prose CoT (regex for number patterns + operators)
2. Translates each equation to a Python expression (lhs = eval-able string, rhs = number)
3. Executes the expression safely via `ast.literal_eval` or restricted `eval`
4. Flags violations where `abs(eval(lhs) - rhs) > tolerance`
Handles: addition, subtraction, multiplication, division, percentages, fractions.
Produces: ExtractionResult(violation_found, expression, computed_value, stated_value, confidence).

Based on Caco (arXiv 2510.04081): code execution validates arithmetic without regex.

**Exp 565 (GPU required):** Run CoACEExtractor on Exp 554's 25 known-incorrect live
responses. Measure TP rate (target: > 0), FP rate, precision, recall. Compare against
VeriCoT (TP=0) and ArithmeticExtractor (TP=0 on IT outputs). If TP > 0, this experiment
GATES Exps 569 and 573.

**Why this matters:** If CoACEExtractor achieves TP > 0 on live incorrect responses,
for the first time the verify-repair pipeline will have a real signal to act on.

### Phase 2: JEPA PURE Redesign — RETRO-060 Critical Path (Exps 566-567)

**Exp 566 (CPU-only):** Implement JEPAPUREMinForm — replace binary BCE loss with the
PURE min-form PRM objective (arXiv 2504.15275). For a reasoning chain of N steps
with score vectors (s_1, ..., s_N):
- correct_chain_score = min(s_t for correct steps)
- incorrect_chain_score = min(s_t for incorrect steps)
- loss = max(0, margin - (incorrect_score - correct_score))

Train on 132-pair FOVER corpus v2. Target: AUC >= 0.700.

**Exp 567 (CPU-only):** Full JEPA v10 retrain with PURE objective on real corpus.
Save to results/jepa_predictor_v10.safetensors. This is the FR-11 mandatory retrain.
Target: AUC >= 0.700 (first time JEPA exceeds random baseline on real data).

### Phase 3: KV260 FPGA Bring-Up (Exp 568, Hardware required)

The Kria KV260 board arrived 2026-04-20. This is the first time hardware is physically
available. Bring-up steps:
1. Boot to Ubuntu image with PYNQ installed
2. Load Ising bitfile from hardware/kv260/ising_sampler_v1.v (synthesize if needed)
3. Set CARNOT_KV260_BITFILE env var
4. Run experiment_313_kv260_bringup.py
5. Benchmark: hardware latency vs CPU (target: < 100μs vs CPU ~358ms JIT overhead)

This unblocks the FPGA acceleration path for all four self-learning tiers.

### Phase 4: Pipeline Integration (Exps 569-570, GPU required, gate on Exp 565 TP>0)

**Exp 569:** Live verify-repair with CoACEExtractor on 50 live GSM8K questions.
Gate: Exp 565 TP_rate > 0. If TP = 0, skip and log blocked_no_extraction.
This is RETRO-033 attempt #11 — but only after extraction is proven to work.
Target: first positive verify-repair number with a functioning extractor.

**Exp 570 (FR-11 mandatory):** Tier 1 self-learning relay with real violations.
Gate: Exp 565 TP_rate > 0. Wire CoACEExtractor into VerifyRepairPipeline +
ConstraintAdditionFromMemory. Track: violations_detected, constraint_added_from_memory,
fp_rate_reduced_across_batches. This closes the FR-11 self-learning loop end-to-end
on REAL constraint fires for the first time in project history.

### Phase 5: New Research (Exps 571-573, CPU-only)

**Exp 571:** HalluField Tier 0e (arXiv 2509.10753). Implement thermodynamic energy-path
hallucination detection from logit distributions. Add as Tier 0e to verification cascade.
Benchmark AUC vs SpilledEnergy (0b) and NUP Probe v4 (0c) on FOVER corpus v2. CPU-only.

**Exp 572:** PRA EBM Beam Search (arXiv 2604.09482). Implement PRA beam search with
EORM as the step-level reward module (K=3 candidates per step, EORM selects minimum
energy). CPU prototype on 20 synthetic arithmetic problems. Validates the energy function
as a generation-time constraint guide, not just post-hoc verifier.

**Exp 573:** Energy-per-Token EORM Calibration (arXiv 2603.20224). Record hardware power
trace via AMD RAPL during 25 GSM8K generations. Correlate with EORM energy at 32-token
boundaries. If r > 0.5, hardware energy becomes a free calibration signal for EORM
training without labeling. CPU-only (RAPL available on AMD Ryzen AI 9 HX 370).

### Phase 6: Retrospective (Exp 574, CPU-only)

Standard milestone retrospective. Track: retro_061_resolved (CoACEExtractor TP>0),
retro_060_resolved (JEPA AUC>0.5), retro_062_resolved (Live 50q A collected),
fpga_alive (KV260 latency < 100μs), fr11_real_violations (Exp 570 violations>0).

---

## Dependency Graph

```
Exp 563 (Live 50q A) ─────────────────────────────────────────┐
                                                               │
Exp 564 (CoACEExtractor impl) ──────────────────────────────┐ │
                                                             │ │
Exp 565 (Live diagnostic, CoACE) ← 564 ─────────── gate ───┼─┘
                              │                             │
                    ┌─────────┴─────────┐                   │
                    ▼                   ▼                   │
           Exp 569 (Live VR)  Exp 570 (FR-11 relay)        │
                                                            │
Exp 566 (PURE margin loss) ─────────────────────────────┐  │
                                                        │  │
Exp 567 (JEPA v10 retrain) ← 566 ──────────────────────┘  │
                                                            │
Exp 568 (KV260 FPGA) ─────── independent ─────────────────┘
                                                            
Exp 571 (HalluField) ─────────── independent (research)
Exp 572 (PRA beam search) ──────── independent (research)
Exp 573 (Energy-per-Token) ─────── independent (research)
                                                            
Exp 574 (Retrospective) ← all
```

---

## Success Criteria

| Criterion | Experiment | Target | Status |
|-----------|-----------|--------|--------|
| retro_062_resolved | Exp 563 | n_pairs >= 40 | Planned |
| retro_061_resolved | Exp 565 | CoACE TP_rate > 0 | Planned |
| retro_060_resolved | Exp 567 | JEPA AUC >= 0.700 | Planned |
| fpga_alive | Exp 568 | latency_us < 100 | Planned |
| live_vr_positive | Exp 569 | signed_improvement > 0 | Gate on 561 |
| fr11_real_violations | Exp 570 | violations_detected > 0 | Gate on 561 |
| hallufield_viable | Exp 571 | AUC > SpilledEnergy baseline | Planned |
| pra_eorm_viable | Exp 572 | violation_rate < greedy | Planned |
| hardware_calibration | Exp 573 | correlation r > 0.5 | Planned |

---

## Hardware Requirements

| Experiment | Hardware | Why | Status |
|-----------|----------|-----|--------|
| Exp 563 | RTX 3090 (CUDA) | Live inference (Gemma4 + Qwen3.5) | Available |
| Exp 564 | CPU | Implementation only | Available |
| Exp 565 | RTX 3090 (CUDA) | Load Exp 554 live responses (need GPU context) | Available |
| Exp 566 | CPU | Training on 132-pair corpus | Available |
| Exp 567 | CPU | JEPA retrain (small model, 132 pairs) | Available |
| Exp 568 | KV260 FPGA | FPGA bring-up | ARRIVED 2026-04-20 |
| Exp 569 | RTX 3090 (CUDA) | 50q live inference + repair | Available |
| Exp 570 | RTX 3090 (CUDA) | Live relay with real violations | Available |
| Exp 571 | CPU | HalluField logit-based (no model load) | Available |
| Exp 572 | CPU | PRA prototype on synthetic data | Available |
| Exp 573 | CPU + RAPL | AMD Ryzen AI 9 HX 370 (RAPL power measurement) | Available |
| Exp 574 | CPU | Retrospective analysis | Available |

---

## Open RETROs Addressed

| RETRO | Title | Action |
|-------|-------|--------|
| RETRO-033 | Live 25q precision benchmark, 10+ attempts | Blocked until RETRO-061 resolved. Exp 569 = attempt #11 AFTER CoACE proven. |
| RETRO-038 | Live 100q VeriCoT+VPRM, 8+ attempts | Same root cause as RETRO-061. Gate on Exp 565 TP>0. |
| RETRO-056 | JEPA AUC below random, two retrains | Exp 567 (PURE objective) targets closure. |
| RETRO-057 | LowRankKAEM energy outside tolerance | Not targeted in .43 — carry to .44 (needs architectural redesign). |
| RETRO-060 | JEPA architecturally anti-correlated | Root cause = binary BCE. Exp 566-567 fix. |
| RETRO-061 | Extraction TP = 0, pipeline no-op | Root cause = format checking. Exp 564-565 fix. |
| RETRO-062 | Live 50q A unrun | Exp 563 fix. |

---

## What Success Looks Like for .43

**Minimum success (must achieve):**
1. CoACEExtractor produces TP > 0 on live IT-model responses (Exp 565)
2. JEPA v10 AUC >= 0.50 with PURE objective (Exp 567)
3. Live 50q A collected (Exp 563)

**Full success:**
1. All minimum criteria met
2. Live verify-repair shows positive improvement with CoACEExtractor (Exp 569)
3. FR-11 closes end-to-end with real constraint violations (Exp 570)
4. KV260 FPGA online (Exp 568)

**What honest_verdict should be:**
- `root_cause_fixed` if CoACE TP > 0 AND JEPA AUC > 0.5
- `partial_fix` if one of the above but not both
- `both_still_blocked` if both critical paths fail again (prompt deeper investigation)

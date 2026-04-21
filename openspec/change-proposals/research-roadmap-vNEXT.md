# Research Roadmap v47 — Milestone 2026.04.47

**Status:** Proposed
**Milestone:** 2026.04.47
**Title:** LLM-as-Extractor Breakthrough — RETRO-070 Resolution and Tier 0c Deployment
**Planned:** 2026-04-21
**Experiments:** 614–626 (13 experiments)

---

## What Milestone 2026.04.46 Proved

Milestone 2026.04.46 ran 13 experiments (601–613) and produced these honest findings:

**Resolved:**
- RETRO-067: ExclusionManifest precheck confirmed working (Exp 601, sentinel file proven)
- RETRO-049: NUP Probe v6 AUC=0.964 — Tier 0c threshold crossed, ready for deployment (Exp 608)
- FR-11 relay: JEPA v12 OOD generalization confirmed (Exp 607, fr11_generalization_confirmed=True)
- D-Wave SamplerBackend wired (Exp 610, dwave_backend_registered=True)

**Still blocked — escalated:**
- RETRO-070 (NEW): Pattern-matching extractor permanently stuck at recall=4%. Fourteen consecutive VR attempts at 0% improvement. Architecture is the problem, not hyperparameters. LLM-as-extractor required before ANY further VR scheduling.
- RETRO-068: CoACEV4 recall=4% (identical to v3). GenPRM-style single-prompt extraction did not improve over regex.
- RETRO-069: DSVD live AUC=0.158 (far below 0.80 threshold). Hidden-state probing trained on synthetic data cannot generalize.
- RETRO-033: Live VR precision — 14 consecutive attempts, 0 positives. Gate closed (gate_open=False).
- RETRO-038: Live 200q Wilson CI — 9 attempts, 0 positives. Same root cause as RETRO-033.

**Research advances:**
- ILV (Interleaved Logic Verification, Exp 606): ilv_recall=0.08 — marginal improvement over regex baseline
- FACT-E (Exp 612): fact_e_no_signal — causal faithfulness probe found no signal on live arithmetic responses
- Synchronous p-bit Ising RTL created (hardware/kv260/ising_sampler_v2.v, area_reduction_estimate ~50%)
- HISR credit assignment wired into D-Wave backend (Exp 610)

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Constraint Extraction Architecturally Broken (RETRO-070, CRITICAL)

**Evidence:** 14 consecutive VR attempts at 0% improvement. CoACEV1 through V4 all failed on live IT-model outputs. Offline recall on hand-crafted test cases: 85–87% (Exp 576). Live recall on real Qwen3.5-0.8B outputs: 4% (Exp 603). The gap is the distribution mismatch between how regex patterns are trained and how IT models write CoT.

**Fix:** LLM-as-extractor (Goal #1b, research-program.md). Use Qwen3.5-0.8B as the extraction LLM, not regex. Two strategies incorporated from arXiv scan:
1. **Structured JSON claim prompt**: "List every arithmetic calculation in this CoT step as JSON array: [{lhs: 'expression', rhs: stated_value}]. Output only JSON." Verify each claim via `eval()`.
2. **SymCode-style code generation** (arXiv 2510.25975): "Write Python that computes: [arithmetic_step]. Output only executable Python." Execute and compare to stated answer.

**Gate:** recall >= 0.20 on 25 known-incorrect live responses. No VR attempt (#15) without crossing this gate.

### Gap 2: DSVD Live Distribution Mismatch (RETRO-069)

**Evidence:** DSVD offline AUC=0.976, live AUC=0.158. Training on synthetic hidden-state pairs produced a model that cannot recognize live Qwen3.5-0.8B violation patterns.

**Fix:** SymCode-style executable verification as the DSVD replacement (arXiv 2510.25975). Instead of probing hidden states (model-specific, distribution-sensitive), generate executable Python from each CoT step and verify by running it. Execution is model-agnostic and distribution-invariant: `eval("47+28") != 76` is always true regardless of how the model phrased the step.

### Gap 3: NUP v6 Ready But Not Deployed; JEPA v13 Needs Calibration

**Evidence:** NUP Probe v6 achieved AUC=0.964 in Exp 608 but has not been wired into VerifyRepairPipeline cascade as the default Tier 0c. JEPA v12 AUC=1.0 on 20 in-distribution pairs is suspicious (overfit) — calibrated training needed.

**Fix:** Wire NUP v6 into cascade (Exp 622). Train JEPA v13 with CAPO calibration loss (arXiv 2604.12632) to prevent overconfidence while maintaining OOD generalization.

---

## Architecture Diagram — Verification Cascade (Post-.47)

```
LLM Response
     │
     ▼
[Tier 0a] CarnotThinkProbe — generative CoT verdict (optional)
     │ verdict != 'incorrect' → continue
     ▼
[Tier 0b] SpilledEnergyDetector — token logit discrepancy
     │ low spill → skip downstream
     ▼
[Tier 0c] NUP Probe v6 ──────── AUC=0.964 (DEPLOYED .47)  ← NEW
     │ score <= threshold → low energy = likely correct, skip
     ▼
[Tier 0d] HallucinationBasinDetector — latent basin depth
     │ deep basin → stable reasoning, skip
     ▼
[Tier 0e] HalluField — thermodynamic instability (advisory)
     │ advisory signal recorded; no short-circuit
     ▼
[Tier 1]  SinkProbe — attention sink concentration
     │ mean_sink_score >= threshold → skip Tier 2+
     ▼
[Tier 2]  EORM — CoT energy reward model (55M params)
     │ energy < threshold → skip Tier 3
     ▼
[Tier 2.5] LLMAsExtractorV1 ── NEW .47 (RETRO-070 fix)     ← NEW
     │ recall >= 20% gate → enable Tier 3
     ▼
[Tier 3]  VerifyRepairPipeline (Ising/KAN)
     │
     ▼
[MetaJuLS] Online extractor weight adaptation (arXiv 2601.00095) ← NEW
```

---

## Phase Descriptions

### Phase 0 — Infrastructure Verification (1 experiment)

**Exp 614: ExclusionManifest Conductor Wire-In Final Validation + DualGPU Prep**

Validates that the exclusion manifest precheck (Exp 601) is actually being consulted before each experiment (RETRO-067). Checks `scripts/conductor_consulted_at.txt` mtime to confirm precheck ran within the last 60s. Also: baseline DualGPU EORM+JEPA parallel forward-pass verification — confirm both GPUs have non-zero utilization when two models are loaded simultaneously. This has been unconfirmed for five consecutive milestones.

Success: `conductor_consulted=True` AND `gpu1_utilization > 0` (DualGPU confirmed working).

### Phase 1 — LLM-as-Extractor (CRITICAL PATH, 3 experiments)

**Exp 615: Live Corpus v3 Expansion (GPU REQUIRED)**

100 more live pairs from GSM8K 350-449 using SOTA models (unsloth/Qwen3.6-35B-A3B-GGUF + unsloth/gemma-4-26B-A4B-it-GGUF if cached, else Qwen3.5-0.8B + Gemma4-E4B-it fallback). Merge into fover_corpus_v5.json. More diverse data improves extractor evaluation and downstream JEPA training.

**Exp 616: LLMAsExtractorV1 — Qwen3.5-0.8B as Extraction LLM (RETRO-070 CRITICAL)**

The fundamental redesign. Three extraction strategies tested on 25 incorrect live responses:
1. **Structured JSON claim prompt**: "List every arithmetic calculation in this CoT step as JSON. Output only JSON: [{\"lhs\": \"expr\", \"rhs\": value}]." Verify each via `eval()`.
2. **SymCode-style code generation** (arXiv 2510.25975): "Write Python that computes the stated answer. Output only Python." Execute and compare.
3. **Step segmentation + eval chain**: Segment CoT by sentence, apply `eval()` to numeric expressions.

Best-recall strategy becomes LLMAsExtractorV1. Target: recall >= 0.20 (5x over v4's 0.04).

**Exp 617: Live Extractor Diagnostic v5 — Gate Verification**

Run best LLMAsExtractorV1 on 25 incorrect + 10 correct live responses. Compute recall, fp_rate, gate_open = recall >= 0.20. If gate_open=True: queue Exp 620. If gate_open=False: queue Exp 623 (TRUST Agents alternative).

### Phase 2 — Model Calibration (2 experiments)

**Exp 618: JEPA v13 CAPO Calibrated Retrain (arXiv 2604.12632)**

JEPA v12 AUC=1.0 on 20 pairs = overfit. Apply CAPO calibration loss: `L_total = L_contrastive + λ * L_calibration` where `L_calibration = |ECE| + MSE(pred_prob, empirical_accuracy)`. Train on fover_corpus_v5.json (300+ live pairs). Target: OOD AUC >= 0.75 AND ECE < 0.10.

**Exp 619: DSVD-SymCode Hybrid — Executable Verification Replacing Hidden-State Probing**

RETRO-069 rethink. Instead of probing hidden states, at each 32-token CoT boundary: (1) use Qwen3.5-0.8B CPU to extract arithmetic operations from partial output; (2) execute via `eval()`; (3) flag mismatches. Distribution-invariant: code execution produces identical results regardless of prose phrasing. Target: live AUC >= 0.50 (vs DSVD's live 0.158).

### Phase 3 — Live VR + Online Adaptation (2 experiments, GATED)

**Exp 620: Live VR Attempt #15 (GATED on Exp 617 gate_open=True)**

If recall >= 0.20, this is the first non-blocked VR attempt in 14 milestones. Run on 25 live questions with LLMAsExtractorV1 extraction + Ising repair. Target: signed_improvement > 0. This is the RETRO-033 resolution gate.

**Exp 621: MetaJuLS Online Adaptation (arXiv 2601.00095)**

Apply MetaJuLS-style meta-RL to adapt the extractor policy weights from the first live batch: update prompt temperature, sampling strategy, and claim confidence threshold based on observed precision/recall. Self-Learning Tier 2 applied to the extractor. Target: extraction policy improves monotonically across batches.

### Phase 4 — Deployment + Hardware (3 experiments)

**Exp 622: NUP v6 Tier 0c Cascade Wire-In**

Wire NUP Probe v6 (AUC=0.964, Exp 608) into `VerifyRepairPipeline` as default Tier 0c. Measure end-to-end cascade latency on 100 synthetic responses. Target: < 5ms average with Tier 0c active.

**Exp 623: TRUST Agents Comparison (arXiv 2604.12184)**

Multi-agent claim extraction as alternative/complement to LLMAsExtractorV1:
- Agent 1: NER prompt to find numeric entities
- Agent 2: construct arithmetic claims from entity pairs
- Agent 3: `eval()` verification

Compare recall vs LLMAsExtractorV1. If TRUST Agents recall > LLMAsExtractorV1: adopt. Runs regardless of Exp 617 gate_open (alternative path if single-pass extraction also fails).

**Exp 624: KV260 Vivado Synthesis v2 + Synchronous Ising Simulation**

Check if Vivado is installed. If yes: synthesize hardware/kv260/ising_sampler_v2.v (synchronous RTL from Exp 612). Capture resource utilization and timing closure report. If no: simulate synchronous Ising v2 in Python to validate correctness. Success: `synthesis_succeeded=True` OR `simulation_validated=True`.

### Phase 5 — Self-Learning + FR-11 + Retrospective (2 experiments)

**Exp 625: Tier 1 Self-Learning Relay — FR-11 Mandatory**

FR-11 mandatory experiment. If Exp 617 gate_open=True: use real violations from Exp 620 to update ConstraintAdditionFromMemory weights. Track per-constraint precision across live corpus (Tier 1 self-learning: upweight constraints catching real errors, downweight noisy ones). If gate closed: run on 25 synthetic violations to maintain FR-11 relay continuity.

**Exp 626: Milestone 2026.04.47 Retrospective**

Read all 13 experiment result files. Compute: retro_070_resolved (recall >= 0.20), retro_033_resolved (signed_improvement > 0), retro_069_resolved (live AUC >= 0.50), fr11_confirmed, nup_v6_deployed. Compare wall time vs .46. Open new RETROs for newly discovered failures.

---

## Dependency Graph

```
614 ─────────────────────────────────────────────────────────┐
615 → 616 → 617 → 620 (GATED gate_open=True) → 621 ────────┤
                 ↘                                           │
                  623 (alternative, runs regardless) ────────┤
618 ──────────────────────────────────────────────────────── ┤
619 ──────────────────────────────────────────────────────── ┤
622 ──────────────────────────────────────────────────────── ┤
624 ──────────────────────────────────────────────────────── ┤
617 → 625 (FR-11 relay uses extractor results if gate_open)─ ┤
All ──────────────────────────────────────────────────────── 626
```

Critical path: 615 → 616 → 617 → 620 → 621

---

## Success Criteria

| Experiment | Gate Condition | RETRO Resolution |
|---|---|---|
| Exp 616 | recall >= 0.20 on 25 incorrect responses | RETRO-070 partial |
| Exp 617 | gate_open=True | Unblocks VR attempt #15 |
| Exp 618 | OOD AUC >= 0.75 AND ECE < 0.10 | JEPA v13 calibrated |
| Exp 619 | live AUC >= 0.50 | RETRO-069 partial |
| Exp 620 | signed_improvement > 0 | RETRO-033 resolved (14 attempts) |
| Exp 622 | cascade_latency < 5ms | NUP v6 deployed |
| Exp 625 | fr11_real_violations_confirmed=True | FR-11 relay maintained |

---

## Hardware Requirements

| Experiment | GPU Required | Est. Duration |
|---|---|---|
| 614 | No (lightweight DualGPU utilization check) | 15 min |
| 615 | Yes (live inference, 2x RTX 3090) | 90 min |
| 616 | No (Qwen3.5-0.8B CPU for extraction) | 30 min |
| 617 | No | 20 min |
| 618 | No (JEPA is tiny, CPU training) | 25 min |
| 619 | No (CPU extraction + eval()) | 25 min |
| 620 | Yes (live VR, 2x RTX 3090, GATED) | 45 min |
| 621 | No | 20 min |
| 622 | No | 20 min |
| 623 | No (CPU inference for extraction LLM) | 30 min |
| 624 | No (synthesis or simulation) | 30 min |
| 625 | No | 20 min |
| 626 | No | 20 min |

**Estimated total:** ~390 min (below per-milestone budget)

---

## Key Papers Incorporated

| Paper | arXiv ID | Experiment |
|---|---|---|
| TRUST Agents multi-agent claim extraction | 2604.12184 | Exp 623 |
| AquaForte LLM-guided SMT | 2601.04675 | Exp 616 (fallback strategy) |
| SymCode neurosymbolic code generation | 2510.25975 | Exp 616, Exp 619 |
| interwhen intermediate verification | 2602.11202 | Filed for .48 |
| CAPO calibration-aware policy optimization | 2604.12632 | Exp 618 |
| MetaJuLS meta-RL constraint propagation | 2601.00095 | Exp 621 |
| ORACLE constraint-led data elicitation | 2603.21140 | Filed for .48 (FOVER v5) |

---

## Open RETROs Carried Forward

| RETRO | Status | Action |
|---|---|---|
| RETRO-033 | Carry — gated on Exp 617 gate_open | VR attempt #15 via Exp 620 |
| RETRO-038 | Carry — gated on RETRO-033 resolution | Scale to 200q after first positive |
| RETRO-057 | Carry — architectural redesign needed | LowRankKAEM redesign (filed .48) |
| RETRO-065 | Carry — RAPL hardware unavailable | Need AMD energy driver |
| RETRO-068 | Resolution attempt via Exp 616 | Close if recall >= 0.20 |
| RETRO-069 | Resolution attempt via Exp 619 | Close if live AUC >= 0.50 |
| RETRO-070 | New — LLM-as-extractor required | Exp 616 primary resolution path |

---

## What NOT to Do in This Milestone

- Do NOT run VR attempt #15 without Exp 617 gate_open=True. Fourteen zero-positive attempts confirm that scheduling VR without a verified extractor wastes the entire experiment budget.
- Do NOT retrain DSVD via hidden-state probing. Exp 604 confirmed the approach fails on live data. Use SymCode-style executable verification instead.
- Do NOT modify scripts/research_conductor.py.
- Do NOT include legacy experiments 308, 260, 309, 425, 410 (in exclusion manifest).

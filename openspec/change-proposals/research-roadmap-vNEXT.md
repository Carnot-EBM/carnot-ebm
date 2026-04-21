# Research Roadmap v48 — Milestone 2026.04.48

**Status:** Proposed
**Milestone:** 2026.04.48
**Title:** interwhen + ORACLE — Mid-Generation Gate Opens, JEPA v14 Calibration, FPGA TCL v2
**Planned:** 2026-04-21
**Experiments:** 627–639 (13 experiments)

---

## What Milestone 2026.04.47 Proved

Milestone 2026.04.47 ran 13 experiments (614–626) and produced these honest findings:

**Resolved:**
- RETRO-069: SymCodeVerifier AUC=0.804 on completed responses (vs DSVD live AUC=0.158). Executable verification decisively beats hidden-state probing. Distribution-invariant: `eval()` is always accurate regardless of model style.
- NUP Probe v6 deployed in cascade (Tier 0c, Exp 622, cascade_latency_ms measured).
- MetaJuLS online adaptation effective (Exp 621, adaptation_effective=True). Extractor policy updates from live batch feedback without retraining.
- JEPA v13 OOD AUC=0.868 (architecture confirmed sound; Exp 618).
- DualGPU: gpu1_utilization confirmed during toy benchmark (Exp 614); however, RETRO-071 opened because no real 13B+ model was tested.

**Still blocked — escalated:**
- RETRO-070 (CRITICAL, carry 2): LLMAsExtractorV1 v1_recall=0.04 — identical to CoACEV4. TRUST Agents even worse (trust_recall=0.0 in Exp 623). Neither architecture improves post-hoc extraction from completed responses. Root cause confirmed: IT models don't write "a + b = c" in completed responses; the violation is buried in prose that no extractor variant can reliably parse after the fact.
- RETRO-033: VR attempt #15 blocked (gate_open=False). Fifteen consecutive attempts at 0% improvement.
- RETRO-038: Live 200q Wilson CI — unresolved, same root cause.
- RETRO-057: LowRankKAEM energy accuracy outside 5% tolerance — architectural redesign required.
- JEPA v13 ECE=0.207 (calibration target <0.10 not met) — ORACLE-labeled training data required.

**Research advances:**
- SymCodeVerifier (Exp 619): executable mid-generation verification prototype (AUC=0.804). This is the architectural primitive that makes interwhen viable for .48.
- KV260 synchronous Ising Python simulation validated (Exp 624). RTL functionally correct. Vivado installation is the only remaining blocker for bitfile synthesis.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Post-hoc Extraction Is Architecturally Dead (RETRO-070, CRITICAL)

**Evidence:** After 15 VR attempts and 5 extractor architectures (CoACEV1-V4, LLMAsExtractorV1, TrustAgents), live recall is stuck at 0-4%. Every approach that processes the COMPLETED response fails because:
1. IT models write "We add 47 apples and 28 oranges for a total of 76 fruits." Not "47+28=76."
2. The violation occurs BEFORE the sentence ends — the error was committed when the model wrote "76", but post-hoc extraction sees only the final "76" in isolation.
3. No regex, no LLM prompt, no multi-agent pipeline can reliably reconstruct the arithmetic claim from natural-language prose at 20%+ recall.

**Fix:** Mid-generation monitoring (interwhen, arXiv 2602.11202). The SymCodeVerifier (AUC=0.804 on completed responses) achieves this recall because it runs executable Python. Applied MID-GENERATION at each sentence boundary, it can catch "76" before the sentence completes, while the preceding "47" and "28" are still in the recent context window. This is architecturally different from post-hoc extraction: the violation is visible in the partial response but not in the completed one.

**Gate for VR #16:** interwhen recall >= 0.20 on 25 known-incorrect responses (Exp 629). No VR attempt without crossing this gate.

### Gap 2: JEPA v13 ECE=0.207 — Calibration Requires Real-Distribution Training Data

**Evidence:** JEPA v13 OOD AUC=0.868 confirms the architecture is sound. But ECE=0.207 (target <0.10) means the model's confidence scores don't reflect true accuracy — a score of 0.80 doesn't mean 80% of predictions are correct. Root cause: all training data (FOVER v1-v4) comes from either synthetic violations or live pairs labeled with a binary correct/incorrect flag, not step-level constraint labels matching the live model's output style.

**Fix:** ORACLE (arXiv 2603.21140) generates step-level constraint-labeled data using syllogistic prompting + symbolic verification. Each training sample gets a per-step label derived from the SAME executable verification that drives SymCodeVerifier — closing the offline/live distribution gap. JEPA v14 trained on ORACLE-labeled FOVER v5 corpus with CAPO calibration loss should achieve ECE < 0.10.

### Gap 3: Two Hardware Blockers Persist — DualGPU Unproven for 13B Models, FPGA Vivado Uninstalled

**Evidence:**
- RETRO-071 (NEW, .47): DualGPU parallel forward-pass has never been confirmed with a model >= 13B spread across both RTX 3090s. Exp 614 tested a toy Linear(10,10) model — not representative of real multi-GPU throughput.
- KV260 FPGA: Vivado not installed since board arrived (Exp 568, 584, 624 all `honest_verdict=vivado_not_installed`). Synchronous Ising RTL (ising_sampler_v2.v) is synthesis-ready but the TCL script still targets v1. Update TCL to v2 and document synthesis steps.

**Fix:**
- Exp 632: Load a real 13B model (device_map explicit: '0':'cuda:0', '1':'cuda:1') and measure GPU-1 sustained utilization > 70% during actual forward passes. This is the only way to confirm DualGPU works.
- Exp 636: Update hardware/kv260/synth_ising.tcl to target ising_sampler_v2.v. Check Vivado installation. Generate updated synthesis documentation.

---

## Architecture Diagram — Verification Cascade (Post-.48)

```
LLM Generation (in progress)
     │
     ├─── Every sentence boundary: SymCodeVerifier (Tier 0f NEW)
     │    [interwhen monitor — arXiv 2602.11202]
     │    violation detected mid-gen? → AdapTrack backtrack + rephrase hint
     │    no violation? → continue generation
     │
     ▼
LLM Response (completed)
     │
     ▼
[Tier 0a] CarnotThinkProbe — generative CoT verdict (optional)
     │
     ▼
[Tier 0b] SpilledEnergyDetector — token logit discrepancy
     │
     ▼
[Tier 0c] NUP Probe v6 — AUC=0.964 (DEPLOYED .47)
     │
     ▼
[Tier 0d] HallucinationBasinDetector — latent basin depth
     │
     ▼
[Tier 0e] HalluField — thermodynamic instability (advisory)
     │
     ▼
[Tier 1]  SinkProbe — attention sink concentration
     │
     ▼
[Tier 2]  EORM — CoT energy reward model (55M params)
     │
     ▼
[Tier 2.5] SymCodeVerifier (post-hoc) — executable verification (AUC=0.804)
     │
     ▼
[Tier 3]  VerifyRepairPipeline (Ising/KAN)
     │
     ▼
[MetaJuLS] Online extractor weight adaptation
     │
     ▼
[JEPA v14] ORACLE-calibrated violation predictor (target OOD AUC>=0.80, ECE<0.10)
```

---

## Phase Descriptions

### Phase 0: Infrastructure (Exp 627) — interwhen Monitor
Build the core mid-generation monitoring infrastructure: InterWhenMonitor class that hooks into a transformers generation loop, calls SymCodeVerifier at each sentence boundary, and returns violation signals.

**Why first:** interwhen is the prerequisite for Exp 629 (gate check) and Exp 630 (VR #16). SymCodeVerifier already exists and achieves AUC=0.804. The monitor wraps it for live generation. No new models needed.

### Phase 1: Data Infrastructure (Exp 628) — ORACLE FOVER v5
Build a step-level constraint-labeled corpus using ORACLE methodology. This corpus drives JEPA v14 training (Exp 631) and provides training data for the interwhen monitor's step-level calibration.

**Why second:** JEPA v14 and calibrated training both depend on this corpus. Run before JEPA training, in parallel with interwhen gate check.

### Phase 2: Gate Check and VR (Exps 629-630) — interwhen Diagnostic + VR #16
Test interwhen monitor on 25 known-incorrect live responses. Gate: recall >= 0.20. If gate opens, run VR attempt #16 immediately with interwhen as the extractor.

**Why gated:** 15 consecutive VR attempts at 0% improvement prove that running VR without a working extractor wastes a full experiment slot. The gate enforces discipline.

### Phase 3: JEPA v14 Calibrated Retrain (Exp 631)
Train JEPA v14 on ORACLE-labeled FOVER v5 corpus with CAPO calibration loss. Target: OOD AUC >= 0.80, ECE < 0.10.

**Why now:** ORACLE corpus from Exp 628 provides real-distribution labels. JEPA v13's ECE=0.207 failure was due to synthetic training data — ORACLE fixes the root cause.

### Phase 4: Hardware + DualGPU (Exps 632, 636) — RETRO-071 + FPGA TCL v2
Two independent hardware experiments in parallel:
1. DualGPU: load a real 13B model across both RTX 3090s, measure GPU-1 utilization (RETRO-071).
2. FPGA TCL v2: update synthesis script for ising_sampler_v2.v, document Vivado steps.

### Phase 5: New Research (Exps 633-635, 637) — HERMES, Multilevel KAN, AdapTrack, RETRO-057
Four independent research experiments:
- Exp 633: HERMES tool-augmented verification adapter (arXiv 2511.18760)
- Exp 634: Multilevel KAN training for KAEMEnergy (arXiv 2603.04827)
- Exp 635: AdapTrack constrained generation + backtrack (arXiv 2510.17376)
- Exp 637: LowRankKAEM full-rank + sparse redesign (RETRO-057 closure attempt)

### Phase 6: Self-Learning Relay (Exp 638) — FR-11 Mandatory
Tier 1 self-learning relay. Real violations from Exp 630 if gate open; synthetic fallback otherwise.

### Phase 7: Retrospective (Exp 639)

---

## Dependency Graph

```
Exp 627 (interwhen) ──────────────────────► Exp 629 (gate check)
                                                    │ gate_open?
                                                    ▼
Exp 628 (ORACLE corpus) ─────────────────► Exp 630 (VR #16) ──► Exp 638 (FR-11)
                         │
                         └───────────────► Exp 631 (JEPA v14)

Exp 632 (DualGPU)  ─── independent
Exp 636 (FPGA TCL) ─── independent
Exp 633 (HERMES)   ─── independent
Exp 634 (Multilevel KAN) ─── independent
Exp 635 (AdapTrack) ─── independent
Exp 637 (LowRankKAEM) ─── independent

All ────────────────────────────────────► Exp 639 (retrospective)
```

---

## Success Criteria Table

| Metric | Target | Blocking? | Experiment |
|--------|--------|-----------|------------|
| interwhen live recall | >= 0.20 on 25 responses | YES — gates VR #16 | Exp 629 |
| VR #16 signed_improvement | > 0 (first positive) | YES — RETRO-033 closure | Exp 630 |
| JEPA v14 OOD AUC | >= 0.80 | No | Exp 631 |
| JEPA v14 ECE | < 0.10 | No | Exp 631 |
| DualGPU GPU-1 utilization | > 70% sustained | No | Exp 632 |
| HERMES step-level recall | > 0.20 | No | Exp 633 |
| KAEMEnergy multilevel speedup | >= 2x fewer epochs | No | Exp 634 |
| AdapTrack repair success | > post-hoc baseline | No | Exp 635 |
| FPGA TCL v2 | Updated for ising_sampler_v2.v | No | Exp 636 |
| LowRankKAEM energy accuracy | within 5% | No | Exp 637 |
| FR-11 relay | mode confirmed | No | Exp 638 |

---

## Hardware Requirements

| Experiment | GPU Required | Notes |
|-----------|-------------|-------|
| Exp 627 | Preferred (can stub) | interwhen monitor runs on live generation |
| Exp 628 | Preferred (can stub) | ORACLE uses LLM generation for step labeling |
| Exp 629 | CARNOT_FORCE_LIVE=1 | Gate check requires live inference |
| Exp 630 | CARNOT_FORCE_LIVE=1 MANDATORY | VR attempt — live GPU required |
| Exp 631 | No | JAX training on CPU acceptable |
| Exp 632 | 2x CUDA GPUs MANDATORY | DualGPU proof requires both RTX 3090s |
| Exp 633 | No (CPU stub) | HERMES adapter, CPU prototype |
| Exp 634 | No | KAN training on CPU |
| Exp 635 | Preferred (can stub) | AdapTrack backtrack test |
| Exp 636 | No | TCL update and Vivado check |
| Exp 637 | No | KAN redesign, CPU |
| Exp 638 | No | Self-learning relay, CPU |
| Exp 639 | No | Retrospective |

---

## RETRO Closure Plan for .48

| RETRO | Current State | .48 Action | Closing Experiment |
|-------|--------------|------------|-------------------|
| RETRO-033 | 15 attempts, 0 positives | VR #16 with interwhen (gated) | Exp 630 |
| RETRO-057 | LowRankKAEM accuracy > 5% tolerance | Full-rank + sparse redesign | Exp 637 |
| RETRO-063 | JEPA calibration ECE=0.207 | ORACLE corpus + CAPO loss | Exp 631 |
| RETRO-066 | CoACE offline/live gap | interwhen closes the gap architecturally | Exp 629-630 |
| RETRO-068 | LLMAsExtractorV1 recall=4% | interwhen mid-generation replaces post-hoc | Exp 627-629 |
| RETRO-070 | Post-hoc extraction architecturally dead | interwhen is the fix | Exp 627-630 |
| RETRO-071 | DualGPU unconfirmed for 13B | Real 13B model split across RTX 3090s | Exp 632 |

**Carry items (not targeted in .48):**
- RETRO-031: Verify closure in result files
- RETRO-038: VR 200q Wilson CI — depends on RETRO-033 first
- RETRO-065: RAPL hardware energy — needs Intel/AMD driver
- RETRO-060: JEPA anti-correlation superseded by JEPA v14 calibration work

---

## New arxiv Findings Incorporated (2026-04-21 Scan)

| Paper | Experiment | Filed For |
|-------|-----------|-----------|
| arXiv 2511.18760 — HERMES tool-augmented verification | Exp 633 | .48 |
| arXiv 2510.17376 — AdapTrack constrained decoding + backtrack | Exp 635 | .48 |
| arXiv 2603.04827 — Multilevel Training for KANs | Exp 634 | .48 |
| arXiv 2601.21210 — Hidden Correctness via Symbolic Verification | Causal checker | .49+ |

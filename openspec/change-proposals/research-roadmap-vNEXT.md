# Research Roadmap — Milestone 2026.04.64

**Title:** JEPA Cross-Domain Fix + Arbiter Calibration + Constraint Accumulation Root Cause

**CalVer:** 2026.04.64 (sequence increment from 2026.04.63)
**Planned Experiments:** Exps 831-842 (12 experiments)
**Date Designed:** 2026-04-25
**Prerequisite:** Milestone 2026.04.63 retro complete (Exp 830)

---

## What Milestone 2026.04.63 Proved

Milestone .63 (Exps 819-830) targeted 10 success criteria. Result: 5 met, 5 missed.

**Wins:**
- Exp 819: injection_field_fixed (discrimination_rate=1.0) — RETRO-ISING-INJECTION-NO-DISCRIMINATION CLOSED
- Exp 820: import_fixed_repair_positive (repair_delta=14, n_baseline_pass=0, n_repair_pass=14) — live GPU code repair runs
- Exp 824: jepa_v23_viable (ood_auc=0.811 on training eval, in_dist_auc=0.870) — FIRST TIME JEPA BREAKS 0.75 TARGET
- Exp 828: probe_viable — Activation Jailbreak Probe working (AUC >= 0.85)
- Exp 829: hf_publish_success — HuggingFace v3 publish complete

**Failures and new diagnostics:**
- Exp 821: constraint_addition_no_delta_live (delta_overall=0.0) — injection fixed but delta still zero
  Root cause unknown: need to audit whether EmbeddingConstraintStore is being populated at all
- Exp 822: arbiter_still_wrong (accuracy_standard=0.17, accuracy_adversarial=0.83) — external field
  fix doesn't help because arbiter uses different code path than IsingConstraintInjector
- Exp 823: blocked_gate — gated on Exp 821 delta>0, which failed
- Exp 824 vs 825 contradiction: Exp 824 shows JEPA v23 OOD AUC=0.811 but Exp 825 shows
  OOD AUC=0.40 (auc_gsm8k=0.36, auc_humaneval=0.76, auc_arc=0.04) — model works on
  training-distribution eval but collapses on deployment test set, especially ARC (0.04).
  Root cause: LIMO corpus was GSM8K/HumanEval-only; zero ARC training data → ARC collapse.
- Exp 826: below_baseline — cross-domain PRM benchmark confirms degradation > 8% AUC
- Exp 827: synthesis_blocked — nextpnr-xilinx not available; iCE40 bitstream still not generated

**9 RETROs open going into .64:**
- RETRO-MANIFEST-FULL-SCOPE — exclusion manifest not applied to all dequeue sites (governance)
- RETRO-SYMCODE-SERIAL — SymCodeVerifier paragraph batching not implemented (5th milestone)
- RETRO-GGUF-CACHE-IMPORT — CLOSED by Exp 820 (import_fixed); retro artifact update pending
- RETRO-ISING-INJECTION-NO-DISCRIMINATION — CLOSED by Exp 819 (retro_injection_closed=true)
- RETRO-ARBITER-FLAT-ENERGY — arbiter_still_wrong despite external field; wrong code path
- RETRO-CONSTRAINT-ZERO-DELTA — constraint_addition_no_delta_live; root cause unknown
- RETRO-TIER1-PLATEAU — blocked by RETRO-CONSTRAINT-ZERO-DELTA
- RETRO-JEPA-OOD — JEPA v23 viable in training eval but ARC collapses in deployment (0.04 AUC)
- RETRO-XILINX-TOOLS-UNAVAILABLE — Vivado/nextpnr-xilinx not installed; iCE40 bitstream pending

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: JEPA Domain Collapse — ARC AUC=0.04 Despite Training Eval 0.811

Exp 824 achieved JEPA v23 OOD AUC=0.811 on the training-domain eval set (FoVer-labeled GSM8K
steps). But Exp 825's deployment eval shows per-domain collapse:
- GSM8K (in-distribution): AUC=0.36 — unexpectedly low (should be >0.80 in-dist)
- HumanEval (OOD): AUC=0.76 — reasonable since Exp 824 included HumanEval pairs
- ARC-Challenge (OOD planning): AUC=0.04 — near-random, zero training signal for planning steps

**Root cause analysis:**
1. Exp 824's LIMO corpus (70 curated pairs) included 0 ARC training examples. The model has
   no representation for planning-style reasoning steps. AUC=0.04 is below random (0.5) —
   the model is systematically wrong on ARC, meaning it learned an anti-ARC bias.
2. The training-eval discrepancy (0.811 vs 0.36 on GSM8K) suggests Exp 824's training evaluation
   used the same data distribution as training, while Exp 825 used a stricter held-out split.
3. A single domain imbalance in training corpus causes systematic failure in deployment.

**Fix:** DG-PRM domain reweighting (arXiv 2507.17849) + DreamPRM per-domain loss weighting
(arXiv 2505.20241) + balanced corpus (20 per domain). The domain reweighting head ensures
the model adapts to domain-specific energy scales at inference time.

### Gap 2: Constraint Accumulation Produces No Energy Delta (Root Cause Unknown)

Exp 821 ran the constraint accumulation pipeline with the external field fix from Exp 819.
delta_overall=0.0 across 3 sessions of 30 GSM8K questions each. This is the same result as
pre-injection-fix, which means the external field injection is not reaching the constraint
accumulation path.

**Root cause hypotheses (to test in Exp 833):**
1. EmbeddingConstraintStore.retrieve() returns empty or near-zero vectors because no constraints
   have been written to the store during the live session (verification pipeline doesn't call
   EmbeddingConstraintStore.write() on detected violations).
2. compute_energy_with_external_field is imported but the VerifyRepairPipeline
   still calls the legacy IsingEBM.energy() method, not the external-field version.
3. The EmbeddingConstraintStore's retrieve() method returns constraints but they
   are zero-padded because the store was initialized empty and session writing
   has a bug (off-by-one, wrong key format, write never called).

Until the root cause is identified, any fix attempt will fail the same way.

### Gap 3: Multi-Agent Arbiter Uses Wrong Energy Path (accuracy_standard=0.17)

The MultiAgentArbiter achieves accuracy_standard=0.17 (below random 0.33) even after the
external field fix. Analysis of Exp 822 results:
- Energies ARE non-zero: [-0.07, +0.14, -0.03] range per scenario
- Arbiter picks lowest energy (index with min energy) → selects LOWEST constraint violation
- But correct answers still have LOW energy 17% of the time → energy is not calibrated

**Root cause:** MultiAgentArbiter.score_agent_outputs() does NOT call
compute_energy_with_external_field. It calls the legacy IsingEBM.energy() directly.
The injection fix in Exp 819 added a new method but didn't update the arbiter code path.

**Additional issue:** The arbiter inverts the energy-correctness relationship in adversarial
scenarios (accuracy_adversarial=0.83) by applying AgentAuditor consensus penalty. This
accidental inversion works for adversarial but hurts standard scenarios. The fix needs to
calibrate absolute energies per-query rather than use consensus penalties as the primary signal.

---

## Architecture Diagram

```
Query
  |
  v
[Tier 0a] CarnotThinkProbe (generative CoT verdict, ThinkPRM arXiv 2504.16828)
  |  fast-path on "incorrect" verdict
  v
[Tier 0b] SpilledEnergyDetector (logit-discrepancy, arXiv 2602.18671)
  |
  v
[Tier 0c] NUP Probe v4 (contrastive energy, AUC=1.0, Exp 523)
  |
  v
[Tier 0d] HallucinationBasinDetector (latent basin depth, arXiv 2604.04743)
  |
  v
[Tier 0e] HalluField (thermodynamic instability, advisory, arXiv 2509.10753)
  |
  v
[Tier 0h] JailbreakDetectionKAN + ActivationJailbreakProbe (Exp 828, AUC >= 0.85)
  |  SAFETY_GATE on jailbreak
  v
[Tier 1]  SinkProbe (attention sink concentration, arXiv 2604.10697)
  |
  v
[Tier 2]  EORM (CoT energy reward model, 55M params)
  |
  v
[Tier 2.1] JEPAReasonerProbe (latent-space reasoning, AUC=0.993, Exp 726)
  |
  v
[Tier 2.5] SymCodeVerifier (executable arithmetic, AUC=0.804 live)
  |         [.64 FIX: paragraph batching for performance — RETRO-SYMCODE-SERIAL]
  v
[Tier 2.6] HermesVerifierAdapter (step-boundary feedback loop)
  |
  v
[Tier 2.7] CausalReasoningVerifier (causal entailment across steps)
  |
  v
[Tier 3]  IsingEBM (full constraint verification, 0.006 ms/check)
  |        + compute_energy_with_external_field [FIXED .63, Exp 819]
  |        + EmbeddingConstraintStore [.64 FIX: root cause diagnosis + schema enforcement]
  |
  v
[Tier 3.5] JEPA v24 DG-PRM (domain-balanced OOD predictor) [.64 FIX: domain reweighting]
  |         TARGET: per-domain min AUC > 0.55, overall OOD > 0.65
  v
MultiAgentArbiter [.64 FIX: route through compute_energy_with_external_field]
  + energy normalization per-query (z-score)
  + calibrated sign convention validation

Self-Learning Loop (FR-11):
  Tier 1: EmbeddingConstraintStore [.64 FIX: write path diagnosis] → IsingEBM external field
  Tier 3: JEPA v24 cascade deploy (gated on OOD > 0.65 per-domain)

VG-Search Scheduling (Exp 815, 50% skip rate active)
KV260 Hardware Path: iCE40 N=32 bitstream generation [.64: icepack run]
```

---

## Phase Descriptions

### Phase 0: Governance Pre-flight (Exp 831)

**Goal:** Update MILESTONE_PREREQS.md with .63 lessons, verify governance, cap experiment count.

**Exp 831 — Governance Pre-flight + RETRO Status Audit (CPU)**
Update MILESTONE_PREREQS.md: add .63 retro-derived immediate actions.
Verify: RETRO-ISING-INJECTION-NO-DISCRIMINATION and RETRO-GGUF-CACHE-IMPORT are CLOSED
(Exps 819/820 verified). Update retro status in ops/status.md.
Check experiment count: 728 experiments vs 700 cap. Report: experiments_over_cap=28.
Mandate: .64 planned cycle MUST be <= 12 new experiments.
honest_verdict=governance_ready or governance_issues.

### Phase 1: Root Cause Diagnostics (Exps 832-833)

**Goal:** Identify the exact code paths responsible for JEPA ARC collapse and constraint delta=0.
Fixes cannot be designed without knowing root causes.

**Exp 832 — JEPA v23 ARC Domain Collapse Diagnosis (CPU)**
Load JEPA v23 weights (results/experiment_824_jepa_v23_limo_corpus.json, checkpoint).
Run JEPA v23 predictions on stratified mini-sets: 10 GSM8K steps, 10 HumanEval steps,
10 ARC-Challenge steps (synthetic/CPU-generated reasoning steps).
For each step: compute JEPA prediction score AND EORM energy AND feature vector.
Analyze: (1) are ARC feature vectors in-distribution relative to training features?
(2) are JEPA scores near 0.5 (uncertain) or near 0.0/1.0 (overconfident wrong)?
(3) what is the variance in JEPA scores per domain?
Finding: ARC steps likely have features that map to a low-energy region of the training
distribution, causing the model to systematically predict "correct" for ARC violations.
Document: feature distribution plot per domain (JSON-serializable stats), diagnosis_finding.
honest_verdict=arc_diagnosis_found or arc_diagnosis_inconclusive.

**Exp 833 — Constraint Accumulation v3 Root Cause Diagnosis (CPU)**
Instrument EmbeddingConstraintStore with full operation logging.
Run the exact pipeline from Exp 821 on 5 GSM8K questions (synthetic, CPU-only, no GPU).
Log every operation: (1) n_constraints_written to store during session,
(2) n_constraints_retrieved per query, (3) embedding vector norms (are they zero?),
(4) compute_energy_with_external_field call count, (5) did VR pipeline call store.write()?
Find the specific line of code where the constraint chain breaks.
Expected finding: VR pipeline detects violations but does not call EmbeddingConstraintStore.write()
— the write path exists but is never invoked during live verification.
honest_verdict=write_path_missing or retrieval_returns_zeros or
external_field_not_called or store_populated_delta_computation_bug.

### Phase 2: Core Fixes (Exps 834-836)

**Goal:** Fix JEPA domain collapse, arbiter calibration, and constraint accumulation.

**Exp 834 — JEPA v24 DG-PRM Domain-Balanced Training (RETRO-JEPA-OOD, CPU)**
Incorporate arXiv 2507.17849 (DG-PRM) and arXiv 2505.20241 (DreamPRM):
1. Build balanced training corpus: 20 GSM8K + 20 HumanEval + 20 ARC-Challenge + 10 SVAMP.
   ARC steps: generate 20 synthetic planning-step pairs from ARC-Challenge questions
   (correct step: "Therefore A is correct because..." vs incorrect: "Therefore B is
   correct because...") with Z3/symbolic verification where possible.
2. DreamPRM per-domain loss weighting: compute per-domain validation loss from Exp 825
   results. ARC weight = 5.0 (worst domain), HumanEval weight = 1.3, GSM8K weight = 1.0.
3. ΔEnergy triplet loss weighting (arXiv 2510.11296): weight each triplet by
   (E_negative - E_positive) magnitude. Large energy gaps get higher loss weight.
4. DG-PRM inference domain head: 4-class softmax (gsm8k/humaneval/arc/svamp).
   At inference: multiply JEPA score by domain_weight[predicted_domain].
5. Train 200 epochs with triplet loss + domain reweighting.
Evaluate: per-domain AUC (gsm8k, humaneval, arc, svamp) + overall OOD AUC.
Target: min(per-domain AUC) > 0.55, overall OOD > 0.65.
honest_verdict=jepa_v24_domain_balanced (all >= 0.55) or
jepa_v24_improvement (overall > 0.65) or jepa_v24_still_unbalanced.

**Exp 835 — Arbiter Energy Calibration Fix v2 (RETRO-ARBITER-FLAT-ENERGY, CPU)**
Root cause identified: MultiAgentArbiter does NOT call compute_energy_with_external_field.
Fix: update MultiAgentArbiter.score_agent_outputs() to call
IsingConstraintInjector.compute_energy_with_external_field() for each agent response.
Additional calibration: z-score normalize energies per arbiter call so relative differences
are amplified (avoid tiny differences like [-0.07, 0.14] determining outcomes).
Also verify sign convention: correct responses must produce E < mean(E_all_agents).
Test: 12 scenarios from Exp 822 (6 standard + 6 adversarial).
Target: accuracy_standard >= 0.67 (2/3 correct on standard scenarios).
honest_verdict=arbiter_calibrated (accuracy_standard >= 0.67) or
arbiter_partial or arbiter_still_wrong.

**Exp 836 — Constraint Accumulation Fix v3 (RETRO-CONSTRAINT-ZERO-DELTA, GPU)**
Based on Exp 833 diagnosis: fix the specific code path that breaks constraint accumulation.
If write_path_missing: add EmbeddingConstraintStore.write() call in VerifyRepairPipeline
after each verified violation, parameterized by enable_constraint_accumulation flag.
If retrieval_returns_zeros: fix EmbeddingConstraintStore initialization to use sentence-
transformers embedding (not zero initialization), verify with retrieval AUC check.
If external_field_not_called: wire compute_energy_with_external_field into the
VerifyRepairPipeline.verify() critical path.
After fix: run 30 GSM8K × 3 sessions on live GPU (CARNOT_FORCE_LIVE=1).
Measure delta_overall = precision_s3 - precision_s1.
Target: delta_overall > 0 AND n_constraints_written_session1 > 0.
honest_verdict=constraint_accumulation_fixed (delta>0) or
write_path_fixed_no_delta (write fixed but delta still 0) or still_blocked.

### Phase 3: Self-Learning Integration (Exps 837-838)

**Goal:** Wire working infrastructure into self-learning relay and deploy JEPA v24.

**Exp 837 — FR-11 Tier 1 Live Relay v3 (FR-11 mandatory, GPU, gated on Exp 836 delta>0)**
Run 5-session self-learning relay on live GPU.
Session length: 15 questions per session (reduced from 30 to fit in 30-min watchdog).
After each session: update EmbeddingConstraintStore with detected violations (write path).
Capacity-constrained update (arXiv 2507.21479): update only top-K=3 highest-variance
constraint types, freeze well-calibrated ones.
Measure: precision_s1, precision_s2, ..., precision_s5. Plot monotonicity.
Tier 1 criterion: precision non-decreasing AND delta_s1_to_s5 > 0.
honest_verdict=tier1_relay_works_live or tier1_plateau_persists_live or blocked_gate.

**Exp 838 — JEPA v24 Multi-Domain Deployment as Tier 3.5 (CPU, gated on Exp 834)**
Read results/experiment_834_jepa_v24_dg_prm.json. If min(per_domain_auc) < 0.55: blocked.
If viable: wire JEPA v24 into ThreeTierPipeline as Tier 3.5 (replace JEPA v23 if present).
Emit VerificationCertificate for each prediction: (step_id, domain_label, jepa_score,
energy_delta, domain_weight, z3_verdict, confidence_score).
Test: 30 held-out steps (10 per domain GSM8K/HumanEval/ARC). Verify:
(1) domain_label matches actual domain (classifier accuracy >= 0.80),
(2) tier35_deployed=True logged to pipeline state.
Update _bmad/traceability.md: FR-11 Tier 3 status = DEPLOYED if tier35_deployed=True.
honest_verdict=jepa_v24_tier35_deployed or jepa_v24_not_deployed_below_gate.

### Phase 4: Hardware + Benchmark (Exps 839-840)

**Goal:** Generate actual iCE40 FPGA bitstream from N=32 synthesis, run credible live benchmark.

**Exp 839 — KV260 iCE40 Bitstream Generation (CPU, gated on Exp 816 synthesis_clean)**
Read results/experiment_816_kv260_synthesis_v2.json. Requires honest_verdict containing
"synthesis_clean" or "lut_count" to be present (synthesis JSON artifact).
Steps:
1. Locate the synthesis JSON artifact from Exp 816 (yosys output).
2. Run nextpnr-ice40 (OSS-CAD-Suite, iCE40 HX8K target):
   nextpnr-ice40 --hx8k --package ct256 --json <synth.json> --asc <output.asc>
3. Run icepack to pack the .asc into a .bin bitstream:
   icepack <output.asc> output/carnot_ising_n32.bin
4. Verify: bitstream header (first 4 bytes) is valid iCE40 magic (0x7EAA997E).
5. Test bitstream simulation if iceprog or simulation tool available.
Deliverable: output/carnot_ising_n32.bin (or blocked artifact if tools fail).
honest_verdict=bitstream_generated or pnr_failed or synthesis_artifact_missing.

**Exp 840 — Live Full Precision Benchmark v3 (GPU, CARNOT_FORCE_LIVE=1)**
50 GSM8K questions with live Qwen3.5-0.8B on GPU.
4 conditions: baseline (no VR), VR-only, VR+JEPA-v24, VR+JEPA-v24+VGSearch.
For each question: record correct/incorrect, inference_mode, constraint_violations_found.
Compute: accuracy_baseline, accuracy_vr, accuracy_full_pipeline.
signed_improvement = accuracy_full_pipeline - accuracy_baseline.
honest_verdict=pipeline_improvement (signed_improvement > 0) or
pipeline_no_improvement or pipeline_degradation.
This is the credible live benchmark after 8 milestones of scaffolding.

### Phase 5: Performance + Retrospective (Exps 841-842)

**Exp 841 — SymCodeVerifier Paragraph Batching (RETRO-SYMCODE-SERIAL, CPU)**
RETRO-SYMCODE-SERIAL (opened .63): SymCodeVerifier processes Exp 627-style multi-paragraph
responses one paragraph at a time with regex, each call taking ~50ms. Batch multiple
paragraphs in a single call: collect all arithmetic expressions from entire response,
evaluate in one exec() call with shared namespace. This eliminates redundant imports and
namespace initialization per paragraph.
Implement: SymCodeVerifier.batch_verify(paragraphs) → SymCodeBatchResult.
Measure: latency_single_para vs latency_batch_10_para. Target: 50% wall-time reduction.
honest_verdict=batching_effective (50% reduction) or batching_marginal or batching_no_gain.

**Exp 842 — Milestone 2026.04.64 Operational Retrospective**
Evaluate all 11 prior experiments. Compute success criteria met/total.
Write improvements_suggested for .65 IMMEDIATE items to MILESTONE_PREREQS.md.
Evaluate: did experiment count stay <= 12 (vs 700 cap)?
Track wall-time trend: regression or improvement vs .63 (3904 min)?
Write results/operational_retro_2026_04_64.json (schema=carnot.operational_retro.v39).

---

## Dependency Graph

```
[Phase 0]
  Exp 831 (governance, CPU)                — no dependency

[Phase 1, diagnostics]
  Exp 832 (JEPA ARC diagnosis, CPU)        — uses Exp 824 checkpoint
  Exp 833 (constraint delta diagnosis, CPU) — instruments live pipeline

[Phase 2, fixes — CPU unless noted]
  Exp 834 (JEPA v24 DG-PRM)               — uses Exp 832 diagnosis + 824 corpus
  Exp 835 (arbiter calibration)            — fixes arbiter code path
  Exp 836 (constraint accumulation fix)    ← gated on Exp 833 root cause; GPU

[Phase 3, integration]
  Exp 837 (FR-11 Tier 1 live relay)       ← gated on Exp 836 delta>0; GPU
  Exp 838 (JEPA v24 deployment)           ← gated on Exp 834 min_domain_auc>0.55

[Phase 4, hardware + benchmark]
  Exp 839 (iCE40 bitstream)               ← gated on Exp 816 synthesis JSON; CPU
  Exp 840 (live benchmark v3)             — GPU; uses JEPA v24 if deployed

[Phase 5, ops]
  Exp 841 (SymCode batching, CPU)         — no dependency
  Exp 842 (retro)                         — reads all prior results
```

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| governance_ready | Exp 831 | MILESTONE_PREREQS.md updated; closed retros confirmed |
| arc_diagnosis_found | Exp 832 | feature distribution analysis complete; finding documented |
| constraint_root_cause_found | Exp 833 | specific code path identified |
| jepa_v24_domain_balanced | Exp 834 | min(per_domain_auc) > 0.55 |
| arbiter_calibrated | Exp 835 | accuracy_standard >= 0.67 |
| constraint_delta_positive | Exp 836 | delta_overall > 0 on live GPU |
| tier1_relay_works_live | Exp 837 | precision non-decreasing across 5 sessions |
| jepa_v24_tier35_deployed | Exp 838 | ThreeTierPipeline Tier 3.5 updated |
| bitstream_generated | Exp 839 | output/carnot_ising_n32.bin valid header |
| pipeline_improvement | Exp 840 | signed_improvement > 0 on live GSM8K |
| batching_effective | Exp 841 | 50% latency reduction for multi-paragraph |

---

## Open RETROs Addressed

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-ISING-INJECTION-NO-DISCRIMINATION | CLOSED (.63 Exp 819) | Audit in Exp 831 |
| RETRO-GGUF-CACHE-IMPORT | CLOSED (.63 Exp 820) | Audit in Exp 831 |
| RETRO-JEPA-OOD | OPEN | Exp 832 (diagnosis) + Exp 834 (DG-PRM fix) |
| RETRO-ARBITER-FLAT-ENERGY | OPEN | Exp 835 (arbiter calibration) |
| RETRO-CONSTRAINT-ZERO-DELTA | OPEN | Exp 833 (diagnosis) + Exp 836 (fix) |
| RETRO-TIER1-PLATEAU | OPEN | Exp 837 (gated on Exp 836) |
| RETRO-SYMCODE-SERIAL | OPEN | Exp 841 (paragraph batching) |
| RETRO-MANIFEST-FULL-SCOPE | OPEN | Human action required (modify conductor) |
| RETRO-XILINX-TOOLS-UNAVAILABLE | OPEN | Exp 839 uses iCE40 path (Vivado not required) |

---

## New Research Papers Incorporated

| Paper | ArXiv | Incorporated In |
|-------|-------|----------------|
| DG-PRM: Dynamic Generalizable PRM | 2507.17849 | Exp 834 (domain reweighting head) |
| DreamPRM: Domain-Reweighted PRM | 2505.20241 | Exp 834 (per-domain loss weighting) |
| ΔEnergy OOD Detection/Generalization | 2510.11296 | Exp 834 (energy-delta triplet weighting) |
| Schema-Constrained Agent Memory | 2604.20117 | Exp 836 (constraint schema enforcement) |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|---------|-------|
| Exps 831-835, 838, 839, 841, 842 | CPU only | JAX_PLATFORMS=cpu |
| Exp 836 | GPU, CARNOT_FORCE_LIVE=1 | Constraint live delta measurement |
| Exp 837 | GPU, CARNOT_FORCE_LIVE=1 | FR-11 Tier 1 live relay |
| Exp 840 | GPU, CARNOT_FORCE_LIVE=1 | Live precision benchmark |
| Exp 839 | CPU + OSS-CAD-Suite | ~/tools/oss-cad-suite/bin |

---

## Key Invariants for .64 Experiments

1. **Domain coverage assertion:** Every JEPA training script MUST have `assert n_arc_pairs >= 10`
   and `assert n_humaneval_pairs >= 10` at startup to prevent domain collapse.

2. **Energy path assertion:** Every arbiter/constraint experiment using energy scoring MUST
   call `compute_energy_with_external_field` not the legacy `IsingEBM.energy()` method.

3. **Write path verification:** Before any constraint accumulation experiment, verify
   EmbeddingConstraintStore.write() is being called: `assert n_constraints_written > 0`
   after the first session, fail loudly with diagnostic details if zero.

4. **Experiment count cap:** The .64 planned cycle MUST not exceed 12 new experiments.
   Report experiments_over_cap in Exp 842 retro.

# Research Roadmap — Milestone 2026.04.63

**Title:** Injection Field Fix + GGUF Unblock + JEPA LIMO Curation + AgentAuditor Arbiter

**CalVer:** 2026.04.63 (sequence increment from 2026.04.62)
**Planned Experiments:** Exps 819-830 (12 experiments)
**Date Designed:** 2026-04-24
**Prerequisite:** Milestone 2026.04.62 retro complete (Exp 818)

---

## What Milestone 2026.04.62 Proved

Milestone .62 (Exps 806-818) targeted 9 success criteria. Result: 4 met, 5 missed.

**Wins:**
- MILESTONE_PREREQS.md gate implemented (Exp 806): CPMI wiring assertion deployed
- OSS-CAD-Suite FPGA toolchain installed (Exp 807): yosys + nextpnr-ice40 + icepack verified
- RETRO-028 CLOSED (Exp 810): Gemma4 OOM fixed via nvidia-smi retry loop; n_valid_responses >= 16
- KV260 N=32 synthesis clean (Exp 816): iCE40 HX8K, 3952 LUTs, 0 synthesis errors
- VG Search scheduling effective (Exp 815): 50% skip rate, 25 Ising calls saved, accuracy preserved

**Failures (direct carry into .63):**
- RETRO-ISING-INJECTION-NO-DISCRIMINATION (Exp 812): injection_negative_delta — energy DECREASES
  when constraint fires. Root cause: coupling matrix diagonal injection is a constant shift,
  not a discriminating signal. Blocks Exps 813 and 814.
- RETRO-ARBITER-FLAT-ENERGY (Exp 817): arbiter_accuracy=0.33; all agent energies=0.0.
  Root cause: downstream of Exp 812 injection failure.
- RETRO-GGUF-CACHE-IMPORT: Exp 811 blocked_model_load_failed due to Python ImportError.
  Gate shift: code repair now blocked by import error, not OOM (RETRO-028 closed).
- RETRO-JEPA-V22-OOD-BELOW-GATE: JEPA v22 ood_auc=0.5 (RA-PRM improved from 0.2).
  Still below 0.75 cascade gate. 11th consecutive failed retrain.
- RETRO-CONSTRAINT-ZERO-DELTA: Exp 813 gated (injection_not_wired). Live constraint delta
  still unmeasured on real GPU.
- RETRO-TIER1-PLATEAU: Exp 814 gated (blocked_no_delta). FR-11 Tier 1 live relay not run.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: IsingEBM Constraint Injection Has Wrong Sign (Blocks 4 Capabilities)

The RETRO-ISING-INJECTION-NO-DISCRIMINATION from Exp 812 is the single most damaging open
issue. It cascaded to block 4 experiments in .62:
- Exp 813 (constraint addition live validation) — injection_not_wired
- Exp 814 (FR-11 Tier 1 live relay) — blocked_no_delta
- Exp 817 (multi-agent arbiter) — all energies 0.0 (arbiter_incorrect)

**Root cause:** IsingConstraintInjector.inject_into_coupling_matrix() adds the constraint
projection to the DIAGONAL of J. In standard Ising E = -0.5 * s^T J s, the diagonal terms
are -0.5 * J[i,i] * s_i^2 = -0.5 * J[i,i] (constant, since s_i^2 = 1 for ±1 spins).
Adding positive values to the diagonal uniformly lowers energy for ALL spin configurations —
it does not discriminate between correct and violated responses.

**Fix:** Replace diagonal injection with external field injection:
E_total = -0.5 * s^T J s - h^T s
Where h = project_to_spin_bias(constraint_embeddings) is a per-spin external field.
When constraint fires: h[i] > 0 and spin s_i = +1 (violation encodes as +1)
→ field_energy = -h[i] * s_i = -h[i] < 0 → TOTAL energy increases by -field_energy > 0.
For correct responses: s_i = -1 → field_energy = -h[i] * (-1) = h[i] > 0 → energy decreases.
Net: violations get HIGHER energy, correct responses get LOWER energy. Discriminating.

**Cascade unlocks (all gated on this fix):**
- Constraint addition live validation (Exp 821)
- Multi-agent arbiter (Exp 822)
- FR-11 Tier 1 live relay (Exp 823)

### Gap 2: JEPA OOD at 0.5 After 11 Consecutive Retrains (FR-11 Tier 3 Still Undeployed)

JEPA has failed to reach OOD AUC >= 0.75 in every retrain since v13. RA-PRM (Exp 809)
improved ood_auc 0.2 → 0.5 by adding retrieval-augmented soft supervision. This is progress,
but insufficient. Root causes:

1. **Training corpus too large and noisy:** 300 FoVer pairs + 750 CPMI triples = 1050 items,
   but the quality is uneven. LIMO (2024) showed 817 curated examples beat 100k random ones for
   LLM training. The same principle applies to JEPA: 50 high-confidence Z3-verified pairs will
   outperform 300 mixed-confidence pairs.

2. **Single-domain training:** All real FoVer pairs come from GSM8K × Qwen3.5-0.8B. OOD eval
   tests different domains. Fix: add 10 Z3-verified HumanEval pairs + 10 SVAMP pairs to make
   the training corpus domain-diverse before the OOD holdout matters less.

3. **Model too small:** JEPAPredictor is a 2-layer MLP. For 11 consecutive failures, the
   architecture may need a structural change: use a contrastive transformer encoder with
   triplet loss rather than binary BCE.

**Fix plan:** Exp 824 implements LIMO-style curation (top-50 pairs by Z3-confidence × CPMI
score) + domain diversity + contrastive transformer architecture. Exp 825 evaluates and
deploys if OOD >= 0.65 (lowered intermediate gate — full 0.75 gate is aspirational after
11 failures; claim Tier 3 if we hit 0.65).

### Gap 3: Code Repair Has Never Produced a Credible Live Positive Result

The strongest Carnot result — code repair (+3.0pp HumanEval from Exp 226) — was from a
simulation artifact (now invalidated). After 5 consecutive milestones attempting live GPU
code repair, RETRO-GGUF-CACHE-IMPORT is the final reported blocker.

RETRO-028 is now CLOSED (Exp 810). The gate shifted: Exp 811 got blocked_model_load_failed
due to a Python ImportError when loading the GGUF loader. This is a 1-2 hour fix: diagnose
the exact ImportError, install/repair the dependency, verify import succeeds, then run 20
HumanEval problems. Even honest_verdict=code_no_improvement would be progress: we need a live
GPU HumanEval result, even if zero improvement.

---

## New Research Incorporated

| Paper | arXiv ID | Incorporation |
|-------|----------|---------------|
| AgentAuditor: Multi-Agent Reasoning Tree Auditing | 2602.09341 | Exp 822 — consensus tie-breaking |
| From Mathematical Reasoning to Code: PRM Generalization | 2506.00027 | Exp 826 — cross-domain benchmark |
| Beyond Outcome Verification: Verifiable PRM | 2601.17223 | Exp 826 — step certificate output |

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
[Tier 0h] JailbreakDetectionKAN (safety gate, AUC=1.0, Exp 775)
  |  [NEW .63] Activation Jailbreak Probe (linear probe, arXiv 2602.11495) — Tier B product
  |  returns SAFETY_GATE if jailbreak detected
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
  |
  v
[Tier 2.6] HermesVerifierAdapter (step-boundary feedback loop)
  |
  v
[Tier 2.7] CausalReasoningVerifier (causal entailment across steps)
  |
  v
[Tier 3]  IsingEBM (full constraint verification, 0.006 ms/check)
  |        + EmbeddingConstraintStore (SPO memory → dynamic constraints)
  |        + IsingConstraintInjector [FIXED .63: external field, not diagonal]
  v
[Tier 3.5] JEPA v23 (OOD predictor, TARGET: AUC >= 0.65) [BLOCKED: v22 AUC=0.5]
  |
  v
MultiAgentArbiter (energy-ranked output selection) [FIXED .63: external field]

Self-Learning Loop (FR-11):
  Verification results → FOVER annotation → JEPA v23 training corpus
  Repair outcomes → EmbeddingConstraintStore (session memory → active constraints)
  EmbeddingConstraintStore → IsingConstraintInjector EXTERNAL FIELD → energy discrimination
  FR-11 Tier 1: precision increase per session (live, gated on external field fix)
  FR-11 Tier 3: JEPA v23 cascade deploy (gated on OOD AUC >= 0.65)
```

---

## Phase Descriptions

### Phase 0: Root Cause Surgery (Exps 819-820) — CPU + GPU

**Goal:** Fix the two root-cause blockers that cascaded into .62 failures before any
downstream experiments run. Both are self-contained fixes requiring under 2 hours each.

**Exp 819 — IsingEBM External Field Fix (RETRO-ISING-INJECTION-NO-DISCRIMINATION, CPU)**
Diagnose why injection_negative_delta occurred in Exp 812. Implement external field energy:
E_total = -0.5 * s^T J s - h^T s where h is the constraint projection.
Add new method: `compute_energy_with_external_field(J, spins, constraint_embeddings)`.
Keep diagonal injection as legacy (ADDITIVE). Test: 10 arithmetic error vs 10 correct responses.
Gate criterion: energy(error) > energy(correct) for >= 8 of 10 pairs.
honest_verdict=injection_field_fixed or injection_still_wrong.

**Exp 820 — GGUF Model Load Diagnostic + Code Repair v5 (RETRO-GGUF-CACHE-IMPORT, GPU)**
Read Exp 811 result JSON to get exact blocked_model_load_failed traceback. Diagnose the
Python ImportError: likely llama-cpp-python package version mismatch or missing CUDA binaries.
Fix the import (pip install / reinstall as needed), verify `from llama_cpp import Llama` succeeds.
Run 20 HumanEval problems. Even code_no_improvement is a valid result — we need a live number.
honest_verdict=import_fixed_repair_positive or import_fixed_no_improvement or still_blocked.

### Phase 1: Cascade Unblock (Exps 821-823) — GPU required, gated on Phase 0

**Goal:** Re-run the three experiments that were blocked by the injection sign error in .62.
All are gated on Exp 819 injection_field_fixed.

**Exp 821 — Constraint Addition Live v2 (RETRO-CONSTRAINT-ZERO-DELTA, gated on Exp 819)**
Re-run Exp 813 with external field IsingConstraintInjector.
30 GSM8K questions × 3 sessions on live GPU. Update EmbeddingConstraintStore per session.
Target: delta_overall > 0. If met: retro_constraint_zero_delta_closed=True.
honest_verdict=constraint_addition_works_live or constraint_addition_no_delta_live.

**Exp 822 — Multi-Agent Arbiter Fix v2 + AgentAuditor Consensus (gated on Exp 819, CPU)**
Re-run Exp 817 with external field energy. Incorporate arXiv 2602.09341 (AgentAuditor):
when all agent energies are within 0.01 of each other (consensus), apply a consensus penalty
that adds energy to agents sharing the majority response pattern.
Test: 6 standard + 6 adversarial scenarios (wrong answer is majority).
honest_verdict=arbiter_correct if accuracy >= 0.80 on all 12 scenarios.

**Exp 823 — FR-11 Tier 1 Live Relay v2 (FR-11 mandatory, gated on Exp 821)**
Re-run Exp 814 with external field injection + live GPU.
5 sessions × 10 questions. Capacity-constrained update (arXiv 2507.21479):
update only top-K=3 highest-variance constraint types per session.
Target: precision non-decreasing sessions 1-5, delta_s1_to_s5 > 0.
honest_verdict=tier1_relay_works_live or tier1_plateau_persists_live.

### Phase 2: JEPA Architecture Overhaul (Exps 824-826) — CPU only

**Goal:** Break the 11-consecutive-retrain failure streak with a fundamentally different
approach: quality over quantity (LIMO principle), domain diversity, and cross-domain benchmark.

**Exp 824 — JEPA v23 LIMO Curated Corpus (RETRO-JEPA-OOD)**
Apply LIMO curation principle: select TOP-50 highest-quality pairs from the full corpus
(FoVer v21_multi + CPMI triples) by Z3 verification confidence × CPMI contrastive score.
Add domain diversity: include 10 Z3-verified HumanEval pairs (from Exp 820 code results) +
10 SVAMP reasoning pairs (synthetic, CPU-generated). Total: 70 curated pairs.
Train JEPA v23 with contrastive triplet loss (not binary BCE): anchor=prefix, positive=correct
step, negative=CPMI hard negative. 100 epochs.
Evaluate: in-distribution AUC + OOD AUC on fover_labeled_steps_live.json.
honest_verdict=jepa_v23_viable (ood_auc >= 0.65) or jepa_v23_improvement (ood_auc >= 0.5)
or jepa_v23_below_random (ood_auc < 0.5).

**Exp 825 — JEPA v23 Cross-Domain Eval + FR-11 Tier 3 Relay**
Evaluate JEPA v23 on 3 domains: GSM8K (in-dist), HumanEval code steps, ARC-Challenge planning.
For each domain: compute per-domain AUC. If overall AUC >= 0.65: wire into ThreeTierPipeline
as Tier 3.5. Emit VerificationCertificate per step: (step_id, energy_delta, constraint_type,
z3_verdict, confidence) — inspired by arXiv 2601.17223 verifiable PRM design.
FR-11 Tier 3 mandatory closure: if tier35_deployed=True, update _bmad/traceability.md.
honest_verdict=jepa_v23_tier35_deployed or jepa_v23_improvement_not_deployed.

**Exp 826 — PRM Cross-Domain Benchmark (arXiv 2506.00027, 2601.17223)**
Using Exp 824-825 results: compute cross-domain degradation vs in-distribution for JEPA v23.
Compare against published PRM transfer baseline (~8% AUC degradation from arXiv 2506.00027).
If Carnot's degradation is larger: diagnose which domain shows largest gap.
Generate VerificationCertificate for 20 failed OOD steps: examine whether Z3/SymCode
corroborates JEPA's high-energy prediction.
honest_verdict=below_baseline or at_baseline or above_baseline.
CPU-only; uses stored CoT step results from Exps 820 + 825.

### Phase 3: Hardware + Safety (Exps 827-828)

**Goal:** Advance the hardware path (KV260 bitstream from iCE40 synthesis) and ship the
Tier B Safety/Jailbreak product.

**Exp 827 — KV260 nextpnr-xilinx Synthesis v3 (gated on Exp 816)**
OSS-CAD-Suite includes nextpnr-xilinx (not just nextpnr-ice40). The KV260 uses Zynq
UltraScale+ FPGA (xczu5eg), not iCE40. Attempt XC Zynq synthesis:
1. Try nextpnr-xilinx --chipdb for xczu5eg. If not supported: fallback to iCE40 HX8K bitstream.
2. For iCE40: run nextpnr-ice40 + icepack to generate actual .bin bitstream from the
   N=32 synthesis JSON produced in Exp 816. Test: icepack produces valid bitstream header.
3. Compare: can the bitstream be loaded on a software-emulated iCE40 via iceprog simulation?
honest_verdict=xilinx_synthesis_clean or ice40_bitstream_generated or synthesis_blocked.

**Exp 828 — Activation Jailbreak Probe (arXiv 2602.11495, Tier B Safety product, CPU)**
Implement linear probe for jailbreak detection on Qwen3.5-0.8B intermediate layer activations.
Load model in eval mode (CPU, no GPU needed). Extract activations at layers [4, 8, 12, 16].
Train sklearn LogisticRegression on 50 JailbreakBench + 50 benign prompts.
Compare: linear_probe_auc vs JailbreakDetectionKAN (Tier 0h, AUC=1.0 from Exp 775).
Measure latency: probe inference must be < 1 ms per query (CPU).
Target: linear_probe_auc >= 0.85, latency < 1 ms.
honest_verdict=probe_viable (>= 0.85 AUC) or probe_partial or probe_not_viable.

### Phase 4: Publishing + Retrospective (Exps 829-830)

**Goal:** Publish validated artifacts to HuggingFace and close the milestone.

**Exp 829 — HuggingFace v3 Publish (Tier A product)**
Publish to huggingface.co/Carnot-EBM:
1. Update 16 existing activation EBM READMEs to clarify Phase 1 research artifact status.
2. Publish JEPA v23 model (if OOD AUC >= 0.65 from Exp 825) with cross-domain benchmark results.
3. Publish IsingConstraintInjector with external field fix as standalone artifact.
4. Update pip install carnot landing README with honest performance summary.
Use SOPS-encrypted HF token from .sops.yaml. Verify upload with huggingface_hub.list_models().
honest_verdict=hf_publish_success or hf_publish_partial or hf_auth_blocked.

**Exp 830 — Milestone 2026.04.63 Operational Retrospective**
Evaluate all 11 prior experiments. Compute success criteria met/total.
Identify new RETROs opened. Confirm RETROs closed.
Write improvements_suggested for .64 with IMMEDIATE items for MILESTONE_PREREQS.md.
Write results/operational_retro_2026_04_63.json with schema=carnot.operational_retro.v38.

---

## Dependency Graph

```
[Phase 0]
  Exp 819 (injection field fix, CPU)  ──────────────┐
  Exp 820 (GGUF import fix, GPU)      ─────┐         │
                                           │         │
[Phase 1, gated]                           │         │
  Exp 821 (constraint live v2) ←────────── │ ────────┘
  Exp 822 (arbiter fix v2)     ←────────────────────┘
  Exp 823 (FR-11 relay v2)     ← gated on Exp 821
                                           │
[Phase 2, CPU independent]                 │
  Exp 824 (JEPA v23 LIMO)     ─────────────┤
  Exp 825 (JEPA v23 eval)     ← gated on 824
  Exp 826 (cross-domain bench) ← uses 820 + 825

[Phase 3, independent]
  Exp 827 (KV260 xilinx, gated on 816)
  Exp 828 (jailbreak probe, CPU independent)

[Phase 4]
  Exp 829 (HF publish, uses 825 model)
  Exp 830 (retro, reads all prior results)
```

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| injection_field_fixed | Exp 819 | energy(error) > energy(correct) >= 8/10 pairs |
| gguf_import_fixed | Exp 820 | live HumanEval result (any honest_verdict) |
| constraint_addition_works_live | Exp 821 | delta_overall > 0 on live GPU |
| arbiter_correct | Exp 822 | accuracy >= 0.80 |
| tier1_relay_works_live | Exp 823 | precision non-decreasing |
| jepa_v23_viable | Exp 824-825 | OOD AUC >= 0.65 |
| cross_domain_at_baseline | Exp 826 | degradation <= 8% AUC vs in-dist |
| bitstream_or_synthesis_clean | Exp 827 | nextpnr-xilinx clean OR iCE40 bitstream |
| probe_viable | Exp 828 | AUC >= 0.85, latency < 1ms |
| hf_publish_success | Exp 829 | models published to HuggingFace |

---

## Open RETROs Addressed by This Milestone

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-ISING-INJECTION-NO-DISCRIMINATION | OPEN | Exp 819 (external field fix) |
| RETRO-ARBITER-FLAT-ENERGY | OPEN | Exp 822 (gated on Exp 819) |
| RETRO-GGUF-CACHE-IMPORT | OPEN | Exp 820 (import diagnostic + fix) |
| RETRO-CONSTRAINT-ZERO-DELTA | OPEN | Exp 821 (gated on Exp 819) |
| RETRO-TIER1-PLATEAU | OPEN | Exp 823 (gated on Exp 821) |
| RETRO-JEPA-OOD | OPEN | Exp 824 (LIMO curation) |
| RETRO-KV260-XILINX | NEW | Exp 827 (nextpnr-xilinx attempt) |

---

## Hardware Requirements

| Phase | Hardware | Purpose |
|-------|----------|---------|
| Phase 0 Exp 819 | CPU | Energy function diagnostics |
| Phase 0 Exp 820 | 2x RTX 3090 GPU, CARNOT_FORCE_LIVE=1 | GGUF inference (GPU 1) |
| Phase 1 Exps 821, 823 | GPU, CARNOT_FORCE_LIVE=1 | Live constraint addition + FR-11 relay |
| Phase 1 Exp 822 | CPU | Arbiter synthetic benchmark |
| Phase 2 Exps 824-826 | CPU | JEPA training + evaluation |
| Phase 3 Exp 827 | CPU | OSS-CAD-Suite synthesis tools |
| Phase 3 Exp 828 | CPU (model in eval mode) | Activation probe |
| Phase 4 Exp 829 | CPU + network | HuggingFace upload |

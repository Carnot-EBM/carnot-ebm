# Research Roadmap — Milestone 2026.04.65

**Title:** JEPA v24b SVAMP Fix + Arbiter Warm-Start + Constraint Retrieval + GGUF Cache + iCE40 N=16

**CalVer:** 2026.04.65 (sequence increment from 2026.04.64)
**Planned Experiments:** Exps 843-854 (12 experiments)
**Date Designed:** 2026-04-25
**Prerequisite:** Milestone 2026.04.64 retro complete (Exp 842)

---

## What Milestone 2026.04.64 Proved

Milestone .64 (Exps 831-842) targeted 11 success criteria. Result: 4 met, 7 missed.

**Wins:**
- Exp 831: governance_ready — RETRO audit complete; RETRO-ISING-INJECTION and RETRO-GGUF-CACHE confirmed closed
- Exp 832: arc_diagnosis_found — JEPA v23 ARC collapse diagnosed: embedding features for ARC steps fall out of training distribution; systematic anti-ARC bias confirmed
- Exp 833: write_path_missing — EmbeddingConstraintStore.write() was never called during live verification; root cause identified precisely
- Exp 836: write_path_fixed — write path wired; 15 constraints stored across 3 sessions (partial: retrieval still broken)
- Exp 841: batching_marginal — SymCodeVerifier paragraph batching implemented; speedup=1.710x; RETRO-SYMCODE-SERIAL CLOSED

**Failures and new diagnostics:**
- Exp 834: jepa_v24_arc_improved_svamp_collapsed — DG-PRM lifted ARC from 0.04 to 0.72 but SVAMP=0.0 (zero training coverage). min_domain_auc=0.0 blocked deployment. Pattern: every domain fix creates a new domain collapse because corpus is not balanced across ALL domains simultaneously.
- Exp 835: arbiter_still_wrong — Z-score normalization did not fix accuracy_standard=0.0. Root cause: Gibbs sampler not warm-started, energies from unconverged MCMC are near-zero magnitude regardless of normalization. Three consecutive milestones non-functional.
- Exp 836: write_path_fixed_no_delta — write fixed (15 constraints stored) but delta_overall=0.0. Retrieval path broken independently: cosine similarity returns near-zero because embeddings are not L2-normalized before storage.
- Exp 837: blocked_gate — gated on Exp 836 delta>0, which failed; FR-11 Tier 1 live relay could not run
- Exp 838: jepa_v24_not_deployed_below_gate — min_domain_auc=0.0 (SVAMP) blocked Tier 3.5 deployment
- Exp 839: pnr_failed — N=32 (3952 LUTs) exceeds iCE40 HX8K P&R budget; synthesis was clean but place-and-route exhausted resources
- Exp 840: simulated_no_verdict — live GPU inference fell back to simulation; no credible benchmark
- Exp 842 retro finding: FIFTH CONSECUTIVE full-milestone wall-time regression (3971 min, +67 vs .63). Slowest-5 identical five consecutive milestones (Exps 786, 527, 491, 627, 603). Experiment count 750 — 50 over the 700 cap. Manifest enforcement still not applied to all dequeue sites.

**9 RETROs open going into .65:**
- RETRO-MANIFEST-FULL-SCOPE — requires human code change (outside experiment scope); provide audit + patch in Exp 843
- RETRO-JEPA-OOD — SVAMP collapse (auc_svamp=0.0); v24b with SVAMP triplets needed
- RETRO-ARBITER-FLAT-ENERGY — Gibbs not warm-started; accuracy_standard=0.0 three milestones
- RETRO-CONSTRAINT-ZERO-DELTA — write fixed; retrieval broken (missing L2-norm)
- RETRO-GGUF-CACHE-IMPORT — carnot/pipeline/gguf_cache.py still missing; 8 consecutive milestone blockade
- RETRO-SVAMP-ZERO-AUC — new .64 RETRO; zero SVAMP coverage in JEPA training corpus
- RETRO-ICE40-PNR-LUT-OVERFLOW — new .64 RETRO; N=32 too large; must reduce to N=16
- RETRO-XILINX-TOOLS-UNAVAILABLE — Vivado not installed; KV260 native synthesis deferred
- RETRO-ISING-INJECTION-NO-DISCRIMINATION — governance audit shows this may still affect the retrieval path; re-validate after Exp 847 fix

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: JEPA Domain Coverage Collapse — SVAMP=0.0 After DG-PRM Fixed ARC

Exp 834 proved DG-PRM reweighting works: ARC AUC improved from 0.04 to 0.72. But SVAMP collapsed to 0.0 because the SVAMP domain had zero training examples. The pattern is clear: every targeted domain fix causes another domain to collapse unless ALL domains are covered simultaneously.

**Root cause:** The JEPA v24 corpus included 0 SVAMP triplets. DreamPRM per-domain reweighting cannot help a domain with zero training signal — it amplifies the existing gradient, not a missing one.

**Fix for .65 (Exp 844 — JEPA v24b):**
1. Add 20 SVAMP triplets to training corpus alongside existing GSM8K/HumanEval/ARC triplets.
2. Assert ALL domains have >= 10 training pairs at script startup before any training begins.
3. Apply DreamPRM per-domain loss weighting (arXiv 2505.20241): compute per-domain validation loss from Exp 834 results; SVAMP weight = 8.0 (worst domain), ARC weight = 1.3 (improved), HumanEval weight = 1.2, GSM8K weight = 1.0.
4. Deploy as Tier 3.5 if min_domain_auc >= 0.50 AND overall_ood >= 0.65.

This is the 11th consecutive JEPA retrain attempt. The pattern of domain-specific failure is now well-understood. The fix is purely a data coverage issue.

### Gap 2: Constraint Retrieval Missing L2-Normalization — Delta=0 Despite Write Path Fix

Exp 836 fixed the write path (15 constraints stored). But delta_overall=0.0 persists because the retrieval path is broken: cosine similarity is computed on un-normalized vectors, producing near-random rankings. The top-retrieved constraint for any query has cosine similarity ~0.1, far too low to produce meaningful energy injection.

**Root cause analysis (to validate in Exp 847):**
1. EmbeddingConstraintStore.write() stores raw sentence-transformer embeddings (L2 norm ~1.0, not exactly 1.0 due to float precision).
2. EmbeddingConstraintStore.retrieve() computes cosine similarity without normalizing the query vector.
3. The cosine similarity threshold (0.7 default) is never crossed, so retrieve() always returns an empty list or near-zero results.
4. IsingEBM external field receives zero-magnitude penalty inputs → energy delta = 0.

**Fix:** L2-normalize all vectors on write AND on query. Verify with retrieval AUROC > 0.80 before running live sessions.

### Gap 3: Arbiter Gibbs Not Warm-Started — Unconverged MCMC Produces Near-Zero Energies

MultiAgentArbiter accuracy_standard=0.0 across three consecutive milestones (.62, .63, .64). Exp 835 applied Z-score normalization and got the same result. The diagnostics show energies_raw in the range [-0.07, +0.14] — tiny magnitudes consistent with an MCMC chain that hasn't mixed.

**Root cause:** IsingEBM Gibbs sampler initializes all spins to zero. At beta=1.0 (the Carnot default), a cold-start from zero takes hundreds of sweeps to reach the energy minimum. The arbiter measures energy after N_sweeps=10 (the default), which is insufficient for convergence. The measured "energy" is mostly initialization noise, not the true Boltzmann distribution.

**Fix (Exp 846 — Gibbs Warm-Start v3):**
Warm-start from the mean-field approximation: `s_i = sign(sum_j J_ij * 0 + h_i) = sign(h_i)`. Run 500 burn-in sweeps before measurement. This is the standard practice in Gibbs sampling (arXiv 2304.06993: warm-start convergence theory) — initialization at the MF fixed point dramatically reduces mixing time. After 500 sweeps from the MF fixed point, energies are at the Boltzmann-distributed equilibrium.

Expected result: energy magnitudes increase from [-0.07, +0.14] range to [-3.0, +3.0] range, making arbitration discriminative. Target: accuracy_standard >= 0.67.

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
[Tier 0f] SemanticEnergyProbe (semantic clustering + Boltzmann, arXiv 2508.14496) [NEW .65]
  |  advisory signal; AUC target > 0.70
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
  |         [.64 FIXED: paragraph batching, speedup=1.710x, RETRO-SYMCODE-SERIAL CLOSED]
  v
[Tier 2.6] HermesVerifierAdapter (step-boundary feedback loop)
  |
  v
[Tier 2.7] CausalReasoningVerifier (causal entailment across steps)
  |
  v
[Tier 3]  IsingEBM (full constraint verification, 0.006 ms/check)
  |        + compute_energy_with_external_field [FIXED .63]
  |        + EmbeddingConstraintStore [write path FIXED .64, retrieval L2-norm .65 FIX]
  |
  v
[Tier 3.5] JEPA v24b (domain-balanced OOD predictor) [.65 FIX: SVAMP triplets added]
  |         TARGET: min_domain_auc >= 0.50, overall_ood >= 0.65

MultiAgentArbiter [.65 FIX: Gibbs warm-start 500 sweeps from sign(h_i)]
  + L2-normalized energy input from EmbeddingConstraintStore
  + Z-score normalization preserved post warm-start
  TARGET: accuracy_standard >= 0.67

Self-Learning Loop (FR-11):
  Tier 1: EmbeddingConstraintStore write + L2-norm retrieval [.65 FIX] → IsingEBM external field
  Tier 3: JEPA v24b Tier 3.5 [.65 target if min_domain_auc >= 0.50]

Hardware Path:
  iCE40 N=16 Ising bitstream [.65: reduce from N=32 → N=16 to fit HX8K P&R budget]
  KV260 native synthesis deferred (Vivado not installed)

GGUF Code Repair Path [.65 UNBLOCK: carnot/pipeline/gguf_cache.py]:
  GGUFCacheResolver → Qwen3.6-35B-A3B-GGUF → SOTA code repair
```

---

## Phase Descriptions

### Phase 0: Governance Pre-flight (Exp 843)

**Goal:** Update MILESTONE_PREREQS.md with .64 lessons, audit RETRO status, produce retirement plan for 50+ over-cap experiments, provide manifest enforcement patch for human review.

**Exp 843 — Pre-flight v14: RETRO Audit + Experiment Retirement Plan (CPU)**
Read results/operational_retro_2026_04_64.json. Identify the 9 open RETROs.
Cross-reference with actual experiment result fields (Exp 836, 839, 841, 842).
Confirm RETRO-SYMCODE-SERIAL is CLOSED (Exp 841 speedup=1.710).
Identify 50+ over-cap experiments (750 actual vs 700 cap).
Produce a retirement_plan.md: list which experiment IDs to retire, sorted by:
  (a) zero residual research value (RETRO closed + experiment re-ran post-closure),
  (b) superseded by newer version (Exp 786 → superseded by Exp 810 outcome),
  (c) hypothesis resolved (diagnostic experiments with confirmed findings).
Provide manifest_enforcement_patch.txt: the exact code changes needed in the dequeue
logic to apply the exclusion manifest at all call sites. This patch is for human review
and application — the experiment does NOT modify research_conductor.py.
MILESTONE_PREREQS.md invariants for .65:
  - assert n_svamp_pairs >= 15 before any JEPA training
  - assert gibbs_warm_start == True before any arbiter energy measurement
  - assert retrieval_l2_normalized == True before any constraint retrieval
  - experiment count: .65 cycle MUST NOT exceed 12 new experiments
honest_verdict: governance_ready if prereqs_updated=True and retirement_plan written.

### Phase 1: JEPA v24b SVAMP-Balanced Training (Exps 844-845)

**Goal:** Fix the SVAMP domain collapse by adding 20 SVAMP triplets to the corpus and retraining with DreamPRM per-domain weighting across all 4 domains simultaneously.

**Exp 844 — JEPA v24b SVAMP-Balanced Domain Training (RETRO-SVAMP-ZERO-AUC + RETRO-JEPA-OOD, CPU)**
Read: results/experiment_834_jepa_v24_dg_prm.json (training checkpoint, corpus paths).
Read: results/experiment_825_jepa_v23_eval_fr11_tier3.json (per-domain AUC baseline).
Read: python/carnot/pipeline/jepa_predictor.py (JEPAPredictor class).

Build corpus v24b:
  - Load existing GSM8K/HumanEval/ARC triplets from Exp 834 corpus.
  - Generate 20 SVAMP triplets: for each SVAMP question, produce a correct reasoning step
    ("The total is X because Y, so the answer is Z.") and an incorrect step
    ("The total is X+1 because Y, so the answer is Z+1.") with symbolic verification
    where the incorrect step fails a numeric equality check.
  - Assert n_svamp_pairs >= 15, n_arc_pairs >= 15, n_humaneval_pairs >= 15, n_gsm8k_pairs >= 15.

DreamPRM reweighting (arXiv 2505.20241):
  - Compute per-domain validation loss from Exp 834 results.
  - SVAMP weight = 8.0 (auc_svamp=0.0 in v24, highest deficit).
  - ARC weight = 1.3 (auc_arc=0.72 in v24, moderate).
  - HumanEval weight = 1.2 (auc_humaneval=0.76 in v24).
  - GSM8K weight = 1.0 (baseline domain).

Train 250 epochs with triplet loss + per-domain weights.
Evaluate: per-domain AUC (gsm8k, humaneval, arc, svamp) + overall OOD AUC.
Target: auc_svamp >= 0.40, min_domain_auc >= 0.50, overall_ood >= 0.65.
honest_verdict:
  - "jepa_v24b_all_domains_viable" if min_domain_auc >= 0.50 AND overall >= 0.65
  - "jepa_v24b_svamp_fixed" if auc_svamp >= 0.40 but min_domain_auc still < 0.50
  - "jepa_v24b_svamp_still_collapsed" if auc_svamp < 0.40

**Exp 845 — JEPA v24b Tier 3.5 Multi-Domain Deployment (CPU, GATED on Exp 844 min_domain_auc >= 0.50)**
Read: results/experiment_844_jepa_v24b_svamp.json.
If min_domain_auc < 0.50: write blocked artifact, document which domain is failing.
If viable: wire JEPA v24b into ThreeTierPipeline as Tier 3.5 (replace JEPA v23 if present).
Emit VerificationCertificate for each prediction: (step_id, domain_label, jepa_score,
  energy_delta, domain_weight, confidence_score, svamp_coverage_flag).
Test: 40 held-out steps (10 per domain: GSM8K/HumanEval/ARC/SVAMP).
Verify: (1) domain_label accuracy >= 0.75, (2) tier35_deployed=True in pipeline state.
Update _bmad/traceability.md: FR-11 Tier 3 status.
honest_verdict: jepa_v24b_tier35_deployed or blocked_below_gate.

### Phase 2: Arbiter Gibbs Warm-Start Fix (Exp 846)

**Goal:** Fix the MCMC convergence issue causing accuracy_standard=0.0 by warm-starting the Gibbs sampler from the mean-field approximation.

**Exp 846 — Multi-Agent Arbiter Gibbs Warm-Start v3 (RETRO-ARBITER-FLAT-ENERGY, CPU)**
Read: results/experiment_835_arbiter_calibration_fix_v2.json (energies_raw, accuracy breakdown).
Read: python/carnot/models/ising.py or equivalent IsingEBM implementation.
Read: python/carnot/pipeline/verify_repair.py (VerifyRepairPipeline IsingEBM usage).

Root cause confirmed: Gibbs sampler initializes s_i = 0. At beta=1.0 with N_sweeps=10 (default),
energy measurement is from an unconverged chain — readings are initialization noise.

Fix: implement GibbsWarmStart protocol in IsingEBM:
  1. Mean-field initialization: s_i = sign(sum_j J_ij * 0 + h_i) = sign(h_i) for each spin i.
     (When bias terms h_i are present, this is the one-step MF approximation.)
     If h_i = 0 for all i (no external bias), fall back to random ±1 initialization.
  2. Burn-in sweep: run 500 Gibbs sweeps after initialization before any energy measurement.
  3. Expose as warm_start_sweeps parameter (default=500, 0=legacy cold-start behavior).

Apply to MultiAgentArbiter.score_agent_outputs():
  - For each agent response, call IsingEBM with warm_start_sweeps=500.
  - Z-score normalize energies per call (preserved from Exp 835).
  - Return energies in [−10, +10] range (instead of [−0.07, +0.14]).

Test: 12 scenarios from Exp 822 (6 standard + 6 adversarial).
Target: accuracy_standard >= 0.67 (4/6 correct on standard scenarios).
Verify: energy magnitude check — abs(mean_energy) > 1.0 across all scenarios.
Spec: add to openspec/capabilities/pipeline/spec.md:
  REQ-SAMPLE-020: GibbsWarmStart MUST initialize from sign(h_i) with burn_in >= 500 sweeps.
  SCENARIO-SAMPLE-030: Arbiter energy measurement; warm-start initialization; energy magnitudes
    in [-10,+10] range (not [-0.07,+0.14]); accuracy_standard >= 0.67.
honest_verdict: arbiter_calibrated if accuracy_standard >= 0.67 or arbiter_partial or arbiter_still_wrong.

### Phase 3: Constraint Retrieval Fix + FR-11 Live Relay (Exps 847-848)

**Goal:** Fix the L2-normalization bug in EmbeddingConstraintStore and validate FR-11 Tier 1 self-learning on live GPU.

**Exp 847 — EmbeddingConstraintStore L2-Norm Retrieval Fix (RETRO-CONSTRAINT-ZERO-DELTA, CPU)**
Read: results/experiment_836_constraint_accumulation_fix_v3.json (n_constraints_written=15, delta=0.0).
Read: python/carnot/pipeline/verify_repair.py or the EmbeddingConstraintStore implementation.
Read: python/carnot/verify/ directory for ConstraintRetriever.

Diagnosis:
  - Print cosine similarities of known stored constraints against known query vectors.
  - Confirm: raw cosine similarities are near 0.0-0.1 without normalization.
  - Confirm: after L2-normalization, same pairs have cosine similarity > 0.7.

Fix:
  1. EmbeddingConstraintStore.write(): L2-normalize embedding before storing.
     embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
  2. EmbeddingConstraintStore.retrieve(): L2-normalize query before computing similarity.
     query = query / (np.linalg.norm(query) + 1e-8)
  3. Lower default cosine threshold from 0.7 to 0.5 (sentence-transformer vectors have
     lower similarity for constraint-type variations than document retrieval).
  4. Add retrieval_l2_normalized=True flag to EmbeddingConstraintStore.__init__.

Validate retrieval quality on held-out constraint pairs:
  - 5 stored constraint types (arithmetic, code, logic, nl, auto).
  - 5 query variations per type.
  - Compute AUROC of retrieval (does the correct type rank highest?).
  - Target: retrieval_auroc > 0.80.
Spec: REQ-VERIFY-150, SCENARIO-VERIFY-175.
honest_verdict: retrieval_fixed if retrieval_auroc > 0.80 or retrieval_partial or retrieval_still_broken.

**Exp 848 — FR-11 Tier 1 Live Relay v4 (FR-11 mandatory, GPU, GATED on Exp 847 retrieval_auroc > 0.70)**
Read: results/experiment_847_constraint_retrieval_l2_fix.json.
If retrieval_auroc <= 0.70: write blocked artifact with diagnostic details.

Run 5-session self-learning relay on live GPU (CARNOT_FORCE_LIVE=1, CARNOT_GPU=1).
Session length: 15 questions per session (Qwen3.5-0.8B, GSM8K).
After each session:
  - EmbeddingConstraintStore.write() with L2-normalized embeddings for each detected violation.
  - Update IsingEBM external field with retrieved constraints for next session.
Track per-session precision: precision_s1, precision_s2, ..., precision_s5.
Capacity-constrained update: update only top-K=5 highest-variance constraint types.
Measure: delta_s1_to_s5 = precision_s5 - precision_s1.
honest_verdict: tier1_relay_works_live if precision non-decreasing AND delta > 0 OR
  tier1_plateau_persists if delta <= 0 OR blocked_gate.
Update _bmad/traceability.md: FR-11 Tier 1 status.

### Phase 4: GGUF Cache Module + SOTA Code Repair (Exps 849-850)

**Goal:** Finally create carnot/pipeline/gguf_cache.py (8th consecutive milestone blockade) and run SOTA code repair with Qwen3.6-35B-A3B-GGUF.

**Exp 849 — GGUF Cache Module Implementation (RETRO-GGUF-CACHE-IMPORT, CPU)**
Read: python/carnot/pipeline/__init__.py — check what is currently exported.
Read: any existing gguf or cache references in the codebase.
Read: openspec/capabilities/pipeline/spec.md — find REQ-PIPELINE-* for caching.

This is a 1-2 hour implementation that has been on the IMMEDIATE list for 8 consecutive milestones.

Implement python/carnot/pipeline/gguf_cache.py:
  - GGUFCacheResolver: resolves a GGUF model path from a model ID.
    Input: model_id (str), quantization (str, default "Q4_K_M"), cache_dir (str, default "models/").
    Output: resolved local path (str) or raises GGUFModelNotFoundError.
  - GGUFModelNotFoundError: raised when model file not found in cache.
  - resolve_gguf_path(model_id, quantization) → str: convenience function.
  - is_gguf_cached(model_id, quantization) → bool: check without resolving.
  - GGUFCacheConfig: dataclass for cache_dir, default_quantization, timeout_s.

Export GGUFCacheResolver and resolve_gguf_path from carnot.pipeline.
Spec: add REQ-PIPELINE-030, SCENARIO-PIPELINE-040 to openspec/capabilities/pipeline/spec.md.
Tests: 100% coverage. Tests use a temp directory as cache_dir to avoid real model downloads.
honest_verdict: gguf_cache_implemented if GGUFCacheResolver exports correctly and tests pass.

**Exp 850 — SOTA GGUF Code Repair v5 (GPU, CARNOT_FORCE_LIVE=1, GATED on Exp 849 gguf_cache_implemented)**
Read: results/experiment_849_gguf_cache_module.json.
Read: any prior SOTA code repair results (Exps 796, 811) for baseline.
Use GGUFCacheResolver to locate Qwen3.6-35B-A3B-GGUF (Q4_K_M) in models/ directory.
If model not cached: write artifact with honest_verdict="model_not_cached", list download command.
If cached:
  25 HumanEval problems in 5 batches of 5. ExperimentTimeoutWatchdog at 60 min.
  Checkpoint per batch. Use llama.cpp loader path (Exp 450 pattern).
  Apply MARS margin gate (arXiv 2601.15498): skip repair for high-margin outputs.
  Baseline: generate once, measure pass@1.
  With repair: generate, verify with CodeExtractor, repair if violations found, re-verify.
  Compute signed_improvement = accuracy_repair - accuracy_baseline.
  honest_verdict: code_repair_positive if signed_improvement > 0 AND n_live >= 15
    or code_repair_negative or model_not_cached.

### Phase 5: Hardware + New Capability (Exps 851-852)

**Goal:** Generate first working iCE40 FPGA bitstream at N=16 and add Semantic Energy as new advisory tier.

**Exp 851 — iCE40 N=16 Ising Bitstream Generation (RETRO-ICE40-PNR-LUT-OVERFLOW, CPU)**
Read: results/experiment_839_kv260_ice40_bitstream.json — P&R failure details, LUT budget.
Read: hardware/kv260/ising_sampler_v3.v or equivalent Verilog.

The N=32 design (3952 LUTs) exceeded the iCE40 HX8K P&R budget (~3500-4000 effective LUTs).
Reducing to N=16 should require approximately N^2/2 = 128 coupling registers + 16 bias registers
= ~140 flipflops, well within HX8K budget.

Steps:
  1. Generate hardware/kv260/ising_sampler_n16.v: parameterize existing Verilog for N=16.
     OR: write a new N=16 Ising sampler using a compact cellular architecture
     (checkerboard Gibbs sweep, 2-spin-per-cycle update, one LUT per conditional).
  2. Synthesize with yosys: use ~/tools/oss-cad-suite/bin/yosys.
     Target: synth_ice40 -top ising_sampler -json output/ising_n16_synth.json
     Expected LUT count: < 1500 (comfortably within HX8K budget of ~7680 total, ~4000 P&R effective).
  3. Place-and-route with nextpnr-ice40:
     nextpnr-ice40 --hx8k --package ct256 --json output/ising_n16_synth.json \
       --asc output/ising_n16.asc
  4. Pack to bitstream: icepack output/ising_n16.asc output/carnot_ising_n16.bin
  5. Verify: first 4 bytes of .bin are iCE40 magic header (0x7EAA997E or implementation-specific).
     If header valid: bitstream_generated=True.
  6. Record: lut_count, fmax_mhz (from timing report), bitstream_size_bytes.
Spec: REQ-FPGA-005 (N=16 bitstream), SCENARIO-FPGA-006.
honest_verdict: bitstream_generated or pnr_failed_n16 (should not happen at N=16) or synthesis_failed.

**Exp 852 — Semantic Energy Probe Tier 0f (arXiv 2508.14496, CPU)**
Read: python/carnot/pipeline/verify_repair.py (cascade tier structure).
Read: python/carnot/models/ising.py or KAN energy scorer.
Reference: arXiv 2508.14496 — "Semantic Energy: Detecting LLM Hallucination Beyond Entropy."

Implement SemanticEnergyProbe as Tier 0f candidate:
  - Input: LLM response text (str), reference_answer (str, optional).
  - Step 1: Extract semantic clusters — split response into declarative sentences.
  - Step 2: Embed each sentence with a lightweight sentence-transformer (or TF-IDF fallback).
  - Step 3: Compute semantic energy: E = -sum_ij exp(-||e_i - e_j||^2 / sigma^2) * coherence_ij
    where coherence_ij = 1 if sentences are semantically consistent, -1 if contradictory.
    (Boltzmann-inspired pairwise energy over sentence cluster.)
  - Step 4: Normalize E by sentence count. High E = sentences diverge (hallucination risk).
    Low E = sentences form a coherent semantic cluster (likely factual).
  - Return: SemanticEnergyResult(energy=float, is_unstable=bool, sentence_count=int, cluster_entropy=float).

Wire as advisory Tier 0f in VerifyRepairPipeline: record is_unstable in VerificationCertificate.
Do NOT short-circuit on Tier 0f (advisory signal only, like HalluField).
Benchmark on 50 synthetic (response, hallucinated/correct) pairs.
Target: AUC_synthetic > 0.70.
Spec: REQ-VERIFY-155 (SemanticEnergyProbe), SCENARIO-VERIFY-180/181.
honest_verdict: probe_viable if AUC > 0.70 or probe_below_threshold.

### Phase 6: Live Benchmark + Retrospective (Exps 853-854)

**Exp 853 — Live Full Precision Benchmark v4 (GPU, CARNOT_FORCE_LIVE=1)**
After all fixes deployed in phases 1-5: run a credible live benchmark.
50 GSM8K questions with live Qwen3.5-0.8B on GPU (CARNOT_FORCE_LIVE=1, CARNOT_GPU=0).
4 conditions:
  (a) baseline: no VR pipeline.
  (b) VR-only: VerifyRepairPipeline with L2-norm retrieval fix.
  (c) VR+JEPA-v24b: with Tier 3.5 if deployed.
  (d) VR+JEPA-v24b+SemanticEnergy: with Tier 0f advisory.
Compute: accuracy_baseline, accuracy_vr, accuracy_full.
signed_improvement = accuracy_full - accuracy_baseline.
Apply apply_env_autofix() to catch CARNOT_FORCE_LIVE=0 at runtime.
ExperimentTimeoutWatchdog(853, timeout_minutes=60). Checkpoint per 10 questions.
If inference falls to simulation: write artifact with honest_verdict="simulated_no_verdict".
honest_verdict: pipeline_improvement if signed_improvement > 0 AND inference_mode="live_gpu"
  or pipeline_no_improvement or pipeline_degradation or simulated_no_verdict.

**Exp 854 — Milestone 2026.04.65 Operational Retrospective**
Standard retro format (schema=carnot.operational_retro.v40).
Read all Exp 843-853 result JSONs.
Compute: total_wall_time_minutes, experiments_completed, avg_time_per_experiment.
Evaluate 11 success criteria (governance_ready, svamp_fixed, jepa_deployed, arbiter_calibrated,
  retrieval_fixed, tier1_relay_works, gguf_cache, code_repair, bitstream_generated,
  semantic_probe, benchmark_improvement).
RETRO audit: update RETRO-SVAMP-ZERO-AUC, RETRO-JEPA-OOD, RETRO-ARBITER-FLAT-ENERGY,
  RETRO-CONSTRAINT-ZERO-DELTA, RETRO-GGUF-CACHE-IMPORT, RETRO-ICE40-PNR-LUT-OVERFLOW status.
Write improvements_suggested for .66 IMMEDIATE items to MILESTONE_PREREQS.md.
Evaluate wall-time trend vs .64 (3971 min).
Write results/operational_retro_2026_04_65.json (schema=carnot.operational_retro.v40).

---

## Dependency Graph

```
[Phase 0]
  Exp 843 (pre-flight v14, CPU)            — no dependency

[Phase 1, JEPA v24b]
  Exp 844 (JEPA v24b SVAMP train, CPU)     — uses Exp 834 checkpoint + Exp 825 domain AUCs
  Exp 845 (JEPA v24b Tier 3.5 deploy)      ← GATED on Exp 844 min_domain_auc >= 0.50

[Phase 2, arbiter]
  Exp 846 (arbiter warm-start v3, CPU)     — fixes IsingEBM in-place; no upstream gate

[Phase 3, constraint retrieval + FR-11]
  Exp 847 (L2-norm retrieval fix, CPU)     — fixes EmbeddingConstraintStore
  Exp 848 (FR-11 Tier 1 relay v4, GPU)    ← GATED on Exp 847 retrieval_auroc > 0.70

[Phase 4, GGUF cache]
  Exp 849 (gguf_cache.py, CPU)             — no dependency
  Exp 850 (SOTA code repair v5, GPU)      ← GATED on Exp 849 gguf_cache_implemented

[Phase 5, hardware + new capability]
  Exp 851 (iCE40 N=16 bitstream, CPU)     — uses ~/tools/oss-cad-suite; no upstream gate
  Exp 852 (Semantic Energy Tier 0f, CPU)   — no dependency

[Phase 6, measurement + ops]
  Exp 853 (live benchmark v4, GPU)         — uses JEPA v24b if deployed; L2-norm fix active
  Exp 854 (retro)                          — reads all prior results
```

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| governance_ready | Exp 843 | MILESTONE_PREREQS.md updated; retirement_plan.md written; manifest_patch provided |
| svamp_corpus_balanced | Exp 844 | n_svamp_pairs >= 15 AND auc_svamp >= 0.40 |
| jepa_v24b_all_domains_viable | Exp 844 | min_domain_auc >= 0.50 AND overall_ood >= 0.65 |
| jepa_v24b_tier35_deployed | Exp 845 | ThreeTierPipeline Tier 3.5 updated; tier35_deployed=True |
| arbiter_calibrated | Exp 846 | accuracy_standard >= 0.67 |
| retrieval_fixed | Exp 847 | retrieval_auroc > 0.80 after L2-norm fix |
| tier1_relay_works_live | Exp 848 | precision non-decreasing AND delta_s1_to_s5 > 0 |
| gguf_cache_implemented | Exp 849 | GGUFCacheResolver exported; tests pass 100% |
| code_repair_positive | Exp 850 | signed_improvement > 0 on live GPU |
| bitstream_generated | Exp 851 | output/carnot_ising_n16.bin valid header |
| semantic_probe_viable | Exp 852 | AUC_synthetic > 0.70 |
| pipeline_improvement | Exp 853 | signed_improvement > 0 on live GSM8K |

---

## Open RETROs Addressed

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-SVAMP-ZERO-AUC | OPEN | Exp 844 (SVAMP triplets + DreamPRM) |
| RETRO-JEPA-OOD | OPEN | Exp 844 (SVAMP fix) + Exp 845 (deployment gate) |
| RETRO-ARBITER-FLAT-ENERGY | OPEN | Exp 846 (Gibbs warm-start v3) |
| RETRO-CONSTRAINT-ZERO-DELTA | OPEN (partial) | Exp 847 (L2-norm retrieval) + Exp 848 (live relay) |
| RETRO-GGUF-CACHE-IMPORT | OPEN | Exp 849 (implement gguf_cache.py) |
| RETRO-ICE40-PNR-LUT-OVERFLOW | OPEN | Exp 851 (N=16 design) |
| RETRO-MANIFEST-FULL-SCOPE | OPEN | Exp 843 provides patch for HUMAN ACTION |
| RETRO-XILINX-TOOLS-UNAVAILABLE | OPEN | Deferred: Vivado not installed; iCE40 path pursued |
| RETRO-ISING-INJECTION-NO-DISCRIMINATION | Audit in Exp 843 | Confirm closed or reopen |

---

## New Research Papers Incorporated

| Paper | ArXiv | Incorporated In |
|-------|-------|----------------|
| Semantic Energy: LLM Hallucination Beyond Entropy | 2508.14496 | Exp 852 (Tier 0f SemanticEnergyProbe) |
| Gibbs Warm-Start Convergence Theory | 2304.06993 | Exp 846 (arbiter warm-start 500 sweeps) |
| DreamPRM: Domain-Reweighted PRM | 2505.20241 | Exp 844 (SVAMP domain reweighting) |
| KANELÉ: KAN for FPGA LUT Evaluation | 2512.12850 | Exp 851 (FPGA synthesis reference) |
| Rethinking Reward Models Multi-Domain | 2510.00492 | Exp 844 (multi-domain evaluation protocol) |
| CHARM: Calibrating Reward Models | 2504.10045 | Exp 846 (post-hoc arbiter calibration) |
| Decomposing Ising Problems on FPGAs | 2602.15985 | Exp 851 (N=16 resource estimation) |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|---------|-------|
| Exps 843-847, 849, 851-852 | CPU only | JAX_PLATFORMS=cpu |
| Exp 848 | GPU 1, CARNOT_FORCE_LIVE=1 | FR-11 relay; GPU 1 thermal advantage |
| Exp 850 | GPU 0, CARNOT_FORCE_LIVE=1 | GGUF inference; Qwen3.6-35B requires ~20GB VRAM |
| Exp 853 | GPU 0, CARNOT_FORCE_LIVE=1 | Live benchmark; Qwen3.5-0.8B |
| Exp 851 | CPU + OSS-CAD-Suite | ~/tools/oss-cad-suite/bin |

---

## Key Invariants for .65 Experiments

1. **SVAMP corpus assertion:** Every JEPA training script MUST have `assert n_svamp_pairs >= 15`
   AND `assert n_arc_pairs >= 15` AND `assert n_humaneval_pairs >= 15` at startup before training.
   This prevents the domain-collapse pattern (ARC fixed → SVAMP collapses) by requiring all
   domains to have representation before training proceeds.

2. **Gibbs warm-start assertion:** Every arbiter energy measurement MUST use warm_start_sweeps >= 500.
   No cold-start from zero. Gate: `assert gibbs.warm_start_sweeps >= 500`.
   Rationale: cold-start MCMC at N_sweeps=10 produces initialization noise, not Boltzmann energies.

3. **L2-normalization assertion:** EmbeddingConstraintStore MUST L2-normalize vectors.
   Gate: `assert retrieval_l2_normalized == True` in EmbeddingConstraintStore.__init__.
   This prevents the cosine similarity = ~0 failure mode that kept delta_overall = 0.0.

4. **Experiment count cap:** The .65 planned cycle MUST not exceed 12 new experiments.
   Report experiments_over_cap in Exp 854 retro. Retirement plan from Exp 843 is the path
   to returning below the 700-experiment cap.

5. **Live inference assertion:** Experiments 848, 850, 853 MUST set CARNOT_FORCE_LIVE=1.
   If inference_mode="synthetic_cpu" is detected: abort with diagnostic artifact.
   Use apply_env_autofix() first in ExperimentTemplate setup.

# Research Roadmap v70 — Milestone 2026.04.70

**Milestone Title:** Iterative Self-Repair + DualGPU Wiring + Self-Learning Tier 4
**CalVer:** 2026.04.70
**Experiments:** Exp 904–916 (13 experiments)
**Previous Milestone:** 2026.04.69 (Exps 892–903, UNRUN — YAML load error caused zero-task milestone)

---

## What Milestone .69 Actually Produced

Milestone .69 suffered a YAML loading error ("Failed to load research-roadmap.yaml: 'title'")
immediately after the milestone was activated. The conductor saw 0 tasks, skipped to the
operational retrospective, and launched the .70 planning agent. **Experiments 892–903 never ran.**

The .69 operational retrospective analyzed cumulative full-milestone efficiency (all 802
experiments across all milestones) and found:

| Metric | Value |
|--------|-------|
| Wall time | 3945 min (-10 min vs .68, -0.25%) |
| Experiments | 802 (102 over 700 cap) |
| Slowest-5 unchanged | UNPRECEDENTED DECUPLE (10th consecutive milestone) |
| Cumulative slow-5 waste | ~2,760 min (46 hours) |
| Documentation-without-application | 11th consecutive milestone |
| DualGPU idle | 4th consecutive post-deploy milestone |
| Retros closed | 0 (4 still open entering .70) |

**What .68 actually proved (the last cycle where experiments ran):**

| Criterion | Result |
|-----------|--------|
| hallusae_retired | ✓ (Exp 880) |
| live_code_repair_positive | ✗ (Exp 881: zero_constraints, signed_improvement=0.0) |
| live_cascade_benchmark | ✓ (Exp 882: inference_mode=live_gpu) |
| vjepa_ood_above_gate | ✓ (Exp 883: ood_auc=0.9211 — massive breakthrough) |
| vjepa_cascade_deployed | ✓ (Exp 884: Tier 2 deployed) |
| spectral_probe_viable | ✓ (Exp 885: AUC=1.0, Tier 0h wired) |
| constrained_decoding_fp_reduction | ✗ (Exp 886: no_fp_reduction) |
| jepa_discriminative_retired | ✓ (Exp 887: VJEPA supersedes discriminative JEPA) |
| fr11_tier3_relay | ✓ (Exp 888: VJEPA-guided constraint addition confirmed) |
| pimi_improved_below_5x | ✗ (Exp 889: sweeps_reduction=4.33, still below 5x) |
| gguf_resolved | ✓ (Exp 890: retired, Qwen/Qwen3.5-0.8B-GGUF repo doesn't exist) |

**Three critical learnings driving .70 design:**

1. **Code repair is blocked by constraint extraction, not the repair algorithm.** Exp 881
   extracted zero constraints from Gemma4-E4B-it responses (`honest_verdict=zero_constraints`).
   ArithmeticExtractor only matches regex patterns (`a + b = c`) that IT models never produce.
   The fix is NOT to improve the extractor — it is to **bypass constraint extraction entirely**
   using iterative execution feedback (arXiv 2604.10508). Run the code, capture the execution
   error, feed it back to Gemma4 for self-repair. Carnot's energy scoring then selects the best
   of N repair attempts. This requires zero extraction — just execute and score.

2. **Experiments 892–903 never ran** due to the YAML loading error. The experiments were well-
   designed (SVAMP diagnosis, VJEPA streaming, KAN Tier 4 seed, DRIFT probe, PIMI final attempt,
   HF publish). Priority experiments are promoted directly into .70.

3. **DualGPU wiring is the highest-ROI governance fix.** DualGPURunner was validated at 1.979x
   speedup (Exp 856) but never wired to the production pipeline. Four consecutive post-deployment
   milestones with both GPUs idle. A 1-2 hour wiring experiment would save ~60 min/milestone
   permanently. This must happen in .70.

---

## Architecture Diagram (entering .70)

```
LLM Response
    │
    ├─► Tier 0a: CarnotThinkProbe (ThinkPRM, generative CoT verify)
    ├─► Tier 0b: SpilledEnergyDetector (logit discrepancy, AUC=0.97)
    ├─► Tier 0c: NUP Probe v4 (bigram contrastive energy, AUC=1.0)
    ├─► Tier 0d: HallucinationBasinDetector (latent basin depth)
    ├─► Tier 0e: HalluField (token-path ensemble variance, AUC=0.97)
    ├─► Tier 0g: StreamingCoT (.66: PHaS trajectory, advisory)
    ├─► Tier 0h: SpectralAttentionProbe (.68: Laplacian eigenvalue, AUC=1.0, advisory)
    ├─► Tier 0i: DRIFTProbe [NEW .70] (multi-layer hidden-state linear probe, advisory)
    ├─► Tier 1: SinkProbe (attention sink concentration)
    ├─► Tier 2: VJEPA v2 (.68: deployed, ood_auc=0.9211)
    │       └─► [.70 NEW path] VJEPA Streaming LogitsProcessor (violation_prob → logit penalty)
    ├─► Tier 2.5: SymCodeVerifier (executable arithmetic, AUC=0.804 live)
    ├─► Tier 2.7: CausalReasoningVerifier (causal entailment, advisory)
    ├─► Tier 2.8: DraftConditionedVerifier [NEW .70] (arXiv 2603.03305, draft-conditioned)
    └─► Tier 3: Ising VerifyRepairPipeline (constraint satisfaction)
           └─► [.70 NEW] IterativeSelfRepair: execute → error feedback → re-generate → energy select

Self-Learning Loop (FR-11):
    Tier 3 violations → Tier 1 Lagrange adaptive weight update (.66) ✓
                     → Forgetting curve decay (.70: Exp 909) [NEW]
    Tier 1 memory → VJEPA training data (.67) ✓
    VJEPA predictions → Constraint addition (.68, Exp 888) ✓
    [.70] Tier 4 → KAN adaptive spline restructuring (Exp 910)

Code Path [.70 NEW]:
    HumanEval problem → Gemma4 generate → Execute code → Capture error
    → Feed error back to Gemma4 → Re-generate → Energy score all attempts
    → Return lowest-energy attempt (Carnot energy selection over self-repair candidates)

SVAMP Path [.70 NEW]:
    Single-step word problem → EstimationVerifier (plausible range bounds)
    → VJEPA v3 retrained on EstimationVerifier-labeled pairs

Hardware:
    iCE40 HX8K: N=8 Ising sampler at 134 LUTs (.66 confirmed) ✓
    iCE40 HX8K: PIMI sparse adjacency final attempt (.70, Exp 914)
    KV260: Vivado still not installed (RETRO-XILINX-TOOLS-UNAVAILABLE, human required)
```

---

## Open RETROs Entering .70

| RETRO | Status | Plan |
|-------|--------|------|
| RETRO-MANIFEST-FULL-SCOPE | OPEN (11 milestones) | Exp 904: final escalation to ops/known-issues.md with HUMAN REQUIRED flag |
| RETRO-SVAMP-ZERO-AUC | OPEN | Exp 907 (diagnosis) → Exp 908 (fix); never ran in .69 |
| RETRO-XILINX-TOOLS-UNAVAILABLE | OPEN | Human required; no .70 action possible |
| RETRO-INERTIA-SWEEPS-TARGET-MISSED | OPEN | Exp 914: PIMI sparse adjacency final attempt (never ran in .69) |

---

## Failed Experiment Rerun Discipline

### Code Repair (Exp 881, 12+ consecutive failures)

**Prior failure record:**
- Exp 881 (zero_constraints): ArithmeticExtractor extracted 0 constraints from Gemma4 responses
- Root cause: IT models produce natural language; regex `a + b = c` never matches
- **Addressed by:** Exp 905 uses iterative execution feedback (arXiv 2604.10508) — no extraction
  needed, code is EXECUTED and errors are fed back to the model
- **retire_if_same_verdict:** true — if Exp 905 still produces zero_improvement on 25q,
  current single-shot code repair approach is retired and the iterative loop is the only path

### PIMI Sweeps (Exp 860, 876, 889 — three prior attempts)

**Prior failure record:**
- Exp 860: EMA only — 2x reduction
- Exp 876: EMA + alpha sweep (checkerboard) — 2x reduction
- Exp 889: Full parallel synchronous updates — 4.33x reduction (pimi_improved_below_5x)
- **Addressed by:** Exp 914 uses copy-node sparsification (arXiv 2503.01177): O(N²) dense
  → O(N·k) sparse couplings; structural hypothesis is orthogonal to update-rule hypothesis
- **retire_if_same_verdict:** true — if sparse v4 achieves <5x, iCE40 PIMI is retired

---

## Phase Descriptions

### Phase 0: Blockers (1 experiment)

**Exp 904: Pre-flight v19** — Audit .69 YAML failure root cause. Attempt manifest enforcement
wiring for the last time — if no path exists without modifying research_conductor.py, write
formal HUMAN REQUIRED flag to ops/known-issues.md. Wire DualGPURunner to VerifyRepairPipeline
if feasible within this experiment's scope. Audit all 4 open RETROs.

### Phase 1: Code Repair — New Approach (2 experiments)

**Exp 905: IterativeSelfRepair v1** (GPU required) — Implement iterative execution-feedback
repair loop: generate → execute → capture error → re-generate (up to 3 attempts) → energy
score all attempts → return best. Uses Gemma4-E4B-it via transformers loader. Run 25 HumanEval
problems. Target: signed_improvement > 0. Based on arXiv 2604.10508 (first 2 rounds capture
most gains). This bypasses ArithmeticExtractor entirely.

**Exp 906: Code Repair 50q Scale-Up** (GPU, gated on Exp 905 signed_improvement > 0) — Scale
to 50 HumanEval problems with 3-attempt iterative repair. Report pass_at_1, signed_improvement.

### Phase 2: SVAMP Diagnosis + Fix (2 experiments)

**Exp 907: SVAMP Root Cause Diagnosis** (CPU) — Run FoVer labeling on 20 SVAMP questions vs
20 GSM8K questions. Measure CoT depth distribution, labeling failure rate. Confirm hypothesis:
SVAMP has mean_cot_depth < 2 (single-step) while GSM8K has > 4 (multi-step). These never ran
in .69.

**Exp 908: SVAMP EstimationVerifier** (CPU, gated on Exp 907 confirmed) — Implement
EstimationVerifier: extract numbers from question, identify operation, compute plausible range,
check if response answer falls in range. Label SVAMP pairs. Retrain VJEPA v3 on SVAMP labels.
Target: svamp_auc > 0.60.

### Phase 3: Self-Learning Tiers (2 experiments)

**Exp 909: Lagrange Forgetting Curve** (CPU) — Add forgetting curve to LagrangeAdaptiveUpdater:
exponential decay w_t = w_0 · exp(-λ · age_t), replay high-violation constraints before expiry.
Based on arXiv 2601.03938 (FOREVER). Benchmark: 10-session relay, measure constraint_precision
with vs without decay. Target: forgetting_improves_precision. Never ran in .69.

**Exp 910: FR-11 Tier 4 KAN Adaptive Spline Seed** (CPU) — Implement KANAdaptiveStructure:
run activation histograms on FoVer corpus, double grid_size for high-activation splines (top 30%),
halve for low-activation (bottom 30%). Uses AutoKnots-style nested spline grid refinement
(arXiv 2412.13423) — preserves learned parameters when adding knots. Train 100 epochs, restructure,
fine-tune 20 more. Target: energy_loss_after < energy_loss_before. Never ran in .69.

### Phase 4: New Probes + New Verifier (2 experiments)

**Exp 911: DRIFT Multi-Layer Ensemble Probe — Tier 0i** (CPU) — Implement DRIFTProbe: extract
hidden states at layers {4,8,12,16}, compute cosine-similarity drift per layer pair, train
linear probe on drift signatures. Enhanced with multi-layer ensemble weighting per arXiv
2604.13386. Target: probe_auc > 0.65. Never ran in .69.

**Exp 912: DraftConditionedVerifier — Tier 2.8** (CPU) — Generate cheap Qwen3.5-0.8B draft
(temperature=0.1, 50 tokens), extract structural constraints from draft, inject into Ising
Tier 3 before run. Based on arXiv 2603.03305. Benchmark: 20 GSM8K questions. Target:
constraint_violation reduction. Never ran in .69.

### Phase 5: Infrastructure + Hardware (2 experiments)

**Exp 913: DualGPU Production Wiring** (CPU/GPU) — Wire DualGPURunner to VerifyRepairPipeline.
The runner was validated at 1.979x throughput (Exp 856) but never used in production. Implementing
parallel GPU dispatch for multi-model benchmark runs will save ~60 min/milestone. Deliverable:
dual_gpu_wired=True, measured throughput vs serial baseline.

**Exp 914: PIMI Sparse Adjacency v4 — FINAL** (CPU + iCE40) — Implement SparsePIMISampler
using copy-node sparsification (arXiv 2503.01177): keep top-k=3 couplings per spin, zero out
rest. Synthesise on iCE40 HX8K. Target: sweeps_reduction >= 5.0 OR pimi_retired verdict to
close RETRO-INERTIA-SWEEPS-TARGET-MISSED. retire_if_same_verdict=true.

### Phase 6: Publishing + Retrospective (2 experiments)

**Exp 915: HuggingFace Publish v3** (CPU) — Publish VJEPA v2 weights (ood_auc=0.9211) to
huggingface.co/Carnot-EBM. Establish IPFS mirror. Update _bmad/architecture.md Last Reconciled
date. Never ran in .69.

**Exp 916: Milestone 2026.04.70 Retrospective** (CPU) — Evaluate all 13 experiments against
success criteria. Compute governance metrics (slowest-5, DualGPU utilisation, retro closures).

---

## Dependency Graph

```
Exp 904 (Preflight)
    │
    ├─► Exp 905 (Iterative Self-Repair 25q, GPU)
    │       └─► Exp 906 (Code Repair 50q, GPU, gated on Exp 905 signed_improvement > 0)
    │
    ├─► Exp 907 (SVAMP Root Cause, CPU)
    │       └─► Exp 908 (SVAMP Estimation Verifier, CPU, gated on Exp 907 confirmed)
    │
    ├─► Exp 909 (Lagrange Forgetting, CPU)  [independent]
    ├─► Exp 910 (KAN Tier 4 Seed, CPU)      [independent]
    │
    ├─► Exp 911 (DRIFT Probe Tier 0i, CPU)  [independent]
    ├─► Exp 912 (Draft Conditioned, CPU)    [independent]
    │
    ├─► Exp 913 (DualGPU Wiring, CPU/GPU)  [independent; critical governance]
    └─► Exp 914 (PIMI Final, CPU+iCE40)    [independent]

Exp 915 (HF Publish, CPU)  [after 884 model path confirmed]
Exp 916 (Retro)             [reads all prior results]
```

---

## Hardware Requirements

| Experiment | GPU | Hardware | Priority |
|------------|-----|----------|----------|
| Exp 905 | RTX 3090 (GPU 0) | CARNOT_FORCE_LIVE=1 | Critical |
| Exp 906 | RTX 3090 (GPU 0) | CARNOT_FORCE_LIVE=1 | Gated on 905 |
| Exp 907-912 | None | CPU only | Any order |
| Exp 913 | Both RTX 3090 | DualGPURunner | Critical governance |
| Exp 914 | None | CPU + iCE40 HX8K | iCE40 toolchain required |
| Exp 915-916 | None | CPU only | Infrastructure |

---

## Success Criteria

| # | Criterion | Gated On | Target |
|---|-----------|----------|--------|
| 1 | manifest_escalated | Exp 904 | enforcement_wired=True OR human_intervention_required documented |
| 2 | code_repair_25q_positive | Exp 905 | signed_improvement > 0 (first in 12+ milestones) |
| 3 | code_repair_50q_positive | Exp 906 | signed_improvement > 0 (scale-up confirmed) |
| 4 | svamp_root_cause_confirmed | Exp 907 | labeling_mismatch_confirmed=True |
| 5 | svamp_auc_above_threshold | Exp 908 | svamp_auc > 0.60 |
| 6 | lagrange_forgetting_improves | Exp 909 | constraint_precision_with_forget > without |
| 7 | kan_tier4_viable | Exp 910 | energy_loss_after < energy_loss_before |
| 8 | drift_probe_viable | Exp 911 | probe_auc > 0.65 |
| 9 | draft_verifier_viable | Exp 912 | constraint_violation reduction > 0 |
| 10 | dual_gpu_wired | Exp 913 | dual_gpu_wired=True AND throughput_ratio > 1.5 |
| 11 | pimi_resolved | Exp 914 | sweeps_reduction >= 5.0 OR pimi_retired (close RETRO) |
| 12 | hf_publish_complete | Exp 915 | publish_confirmed=True AND ipfs_mirror_confirmed |

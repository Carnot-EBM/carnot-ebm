# Research Roadmap v71: Math Repair + Lagrange Closure + R-PRM Tier 2.9 + Self-Learning Tier 2 Code

**Milestone:** 2026.04.71
**Planned:** 2026-04-26
**Predecessor:** 2026.04.70 (11/12 criteria met, 36.9 min total)
**Experiments:** Exps 917–928 (12 experiments)

---

## What Milestone .70 Proved

Milestone .70 was the most successful milestone in the project's history by criteria density:
11/12 criteria in 36.9 wall minutes (vs project average ~300 min+).

**Key breakthroughs:**

- **IterativeSelfRepair works**: Code repair pass rate 8% → 80% (+72pp) on 50 HumanEval problems
  using execute-feedback-retry (arXiv 2604.10508). First positive code repair result after 12
  consecutive blocked milestones.
- **EstimationVerifier works**: SVAMP AUC improved from 0.125 to 0.9 (+0.775) using arithmetic
  range checking. Closes RETRO-SVAMP-ZERO-AUC.
- **KAN Tier 4 seeded**: AutoKnots adaptive spline refinement validated in synthetic setting.
  Structural self-improvement architecture is live.
- **DraftConditioned Tier 2.8 viable**: Structural constraint injection improves Ising solve quality.
- **DRIFTProbe Tier 0i marginal**: Multi-layer drift probe AUC above baseline but below 0.65 threshold.
- **DualGPU wired**: ThreeTierPipeline dispatch connected but throughput not quantitatively confirmed.
- **PIMI research RETIRED**: All three strategies maxed at 4.33x < 5x; scope added to exclusion manifest.
- **RETRO-SVAMP-ZERO-AUC CLOSED**, **RETRO-INERTIA-SWEEPS-TARGET-MISSED RETIRED**.

**Remaining gaps:**

- RETRO-LAGRANGE-ENTROPY-DEGENERATE: Exp 909 failed due to single-constraint corpus where entropy
  is always 0. Root cause identified — multi-constraint corpus will address it.
- DualGPU throughput not quantitatively confirmed (structural wiring only).
- HF publish blocked by SOPS auth injection.
- DriftProbe only marginal — multi-layer ensemble needed.
- IterativeSelfRepair validated for code only; math repair (GSM8K) not yet attempted.

---

## Architecture Diagram (Verification Pipeline)

```
User prompt → LLM → raw response
                          ↓
           [Tier 0a] CarnotThinkProbe (fast CoT verdict)
                          ↓ (uncertain → continue)
           [Tier 0b] SpilledEnergyDetector (logit discrepancy)
                          ↓ (high spill → continue)
           [Tier 0c] NUPProbeV4 (bigram contrastive, AUC=1.0)
                          ↓ (low score → continue)
           [Tier 0d] HallucinationBasinDetector (basin depth)
                          ↓ (shallow basin → continue)
           [Tier 0e] HalluField (thermodynamic instability, advisory)
                          ↓
           [Tier 1] SinkProbe (attention sink concentration)
                          ↓ (uncertain → continue)
           [Tier 2] VJEPA v2 (CoT violation prediction, AUC=0.9211)
                          ↓ (energy > threshold → continue)
           [Tier 2.5] SymCodeVerifier (arithmetic execution, AUC=0.804)
                          ↓
           [Tier 2.6] HermesVerifierAdapter (step-boundary feedback)
                          ↓
           [Tier 2.7] CausalReasoningVerifier (causal entailment)
                          ↓
       NEW [Tier 2.8] DraftConditionedVerifier (structural constraints)  ← Exp 927 wires this
       NEW [Tier 2.9] R-PRM Step Reward (reasoning-driven, arXiv 2503.21295) ← Exp 924 adds this
                          ↓
           [Tier 3] Ising VerifyRepairPipeline (full constraint verify)
                          ↓
           [Repair] IterativeSelfRepair (execute-feedback-retry)
                          ↓
          certificate + repaired response
```

**Self-learning loop (FR-11):**
```
Tier 1 violations → LagrangeMultiplierUpdater [Exp 918 adds forgetting curve]
                 → ConstraintTemplateLibrary [Exp 926 adds code domain patterns]
                 → FoVer → VJEPA v2 retraining [Exp 925 real data KAN Tier 4]
```

---

## Three Biggest Gaps (PRD → Current State)

### Gap 1: Math Reasoning Repair Still Untested

The code repair breakthrough (+72pp on HumanEval) uses Python execution as the oracle. The same
approach can be applied to arithmetic: parse CoT steps to Python expressions, run `eval()`, compare
to LLM's stated answer. If they diverge, feed the discrepancy back. This is the original goal of
the project — verifying and repairing LLM math reasoning. Exps 919-920 address this gap.

### Gap 2: Self-Learning Tier 1 Not Validated in Multi-Constraint Regime

Lagrange forgetting (Exp 909) failed because a single-constraint corpus has entropy = 0 regardless
of decay (p = 1.0 always → -p*log(p) = 0). In production, there are 8+ constraint types with
heterogeneous violation rates. Exp 918 re-runs with a multi-constraint corpus where entropy is
non-degenerate. This closes RETRO-LAGRANGE-ENTROPY-DEGENERATE.

### Gap 3: New Verification Tiers Need Pipeline Integration and Scale Validation

DraftConditioned (Tier 2.8), R-PRM Step Reward (Tier 2.9), and DRIFTProbe are viable in isolation
but not integrated into the main cascade. Exps 924, 927 wire these tiers. Exp 923 improves
DRIFTProbe AUC from marginal to viable via multi-layer ensemble.

---

## Phase Descriptions

### Phase 0: Pre-flight + Governance (Exp 917) — CPU, 20 turns

Audit .70 results, triage 3 open RETROs, set .71 gates. Confirm PIMI retirement in exclusion
manifest. Document RETRO-LAGRANGE-ENTROPY-DEGENERATE root cause and Exp 918 as targeted fix.

### Phase 1: Self-Learning Tier 1 Closure (Exp 918) — CPU, 30 turns

Re-run Lagrange forgetting with 8-constraint heterogeneous corpus where entropy is non-degenerate.
Gate: retire_if_same_verdict=false (root cause identified as corpus design, not algorithm).

### Phase 2: Math Repair Expansion (Exps 919-920) — GPU + CPU, 50+40 turns

Exp 919: MathIterativeSelfRepair on 25 GSM8K questions. Use Python eval() as arithmetic oracle.
Exp 920: Combined EstimationVerifier + MathIterativeSelfRepair pipeline. GATED on Exp 919.

### Phase 3: Infrastructure Closure (Exps 921-922) — GPU + CPU, 30+20 turns

Exp 921: DualGPU throughput benchmark — quantify actual speedup (target: >= 1.7x observed).
GATED on Exp 913 dualgpu_wired. Exp 922: HF Publish v4 with SOPS auth injection.

### Phase 4: Probe + Tier Improvements (Exps 923-925) — GPU + CPU, 40+40+40 turns

Exp 923: DRIFTProbe Ensemble — multi-layer weighted ensemble (arXiv 2604.13386) to improve from
marginal → viable (AUC > 0.65 target). Prior failure addressed by ensemble vs single-layer.
Exp 924: R-PRM Step Reward Tier 2.9 (new paper, arXiv 2503.21295, +11.9 F1 on ProcessBench).
Exp 925: KAN Tier 4 real data training — train AutoKnots KAN on FoVer-labeled real violations.

### Phase 5: Self-Learning Tier 2 + Pipeline Integration (Exps 926-927) — CPU, 40+30 turns

Exp 926: FR-11 Tier 2 code domain memory. Self-learning MANDATORY experiment. Extend CaseMemory
to code repair patterns from Exp 905/906 logs. Accumulate (error_type, repair_pattern) pairs.
Exp 927: DraftConditioned Tier 2.8 integration into ThreeTierPipeline cascade. Measure skip rate.

### Phase 6: Retrospective (Exp 928) — CPU, 20 turns

Evaluate 12 success criteria. Compute wall time. Update ops/status.md.

---

## Dependency Graph

```
Exp 917 (pre-flight)
   │
   ├─→ Exp 918 (Lagrange multi-constraint) [independent, CPU]
   │
   ├─→ Exp 919 (MathIterativeSelfRepair) [GPU]
   │      └─→ Exp 920 (Combined pipeline) [CPU, GATED on 919]
   │
   ├─→ Exp 921 (DualGPU throughput) [GPU, GATED on 913 wired]
   │
   ├─→ Exp 922 (HF Publish v4 SOPS) [CPU, independent]
   │
   ├─→ Exp 923 (DRIFTProbe ensemble) [GPU, prior 911 marginal]
   ├─→ Exp 924 (R-PRM Tier 2.9) [CPU, new paper]
   ├─→ Exp 925 (KAN Tier 4 real data) [CPU, prior 910 seed]
   │
   ├─→ Exp 926 (FR-11 Tier 2 code domain) [CPU, mandatory self-learning]
   ├─→ Exp 927 (DraftConditioned integration) [CPU, prior 912 viable]
   │
   └─→ Exp 928 (Retro) [depends on all]
```

---

## Hardware Requirements

| Experiment | Requires | Why |
|------------|----------|-----|
| Exp 919 MathIterativeSelfRepair | GPU (RTX 3090) | Qwen3.6-35B-A3B-GGUF inference |
| Exp 921 DualGPU throughput | Both RTX 3090s | Actual dual-GPU benchmark |
| Exp 923 DRIFTProbe ensemble | GPU (RTX 3090) | Hidden state extraction |
| All others | CPU only | Python eval, JAX, small models |

---

## Success Criteria for .71

| # | Criterion | Experiment | Target |
|---|-----------|------------|--------|
| 1 | lagrange_entropy_validated | Exp 918 | signed_entropy_improvement > 0 |
| 2 | math_repair_working | Exp 919 | signed_improvement > 0 |
| 3 | combined_pipeline_viable | Exp 920 | combined_auc > individual_max |
| 4 | dualgpu_throughput_confirmed | Exp 921 | observed_speedup >= 1.7 |
| 5 | hf_published | Exp 922 | honest_verdict != "skipped" |
| 6 | drift_probe_viable | Exp 923 | ood_auc_drift > 0.65 |
| 7 | rppm_tier29_viable | Exp 924 | auc > baseline_auc |
| 8 | kan_tier4_real_data | Exp 925 | real_auc > synthetic_auc |
| 9 | tier2_code_memory_works | Exp 926 | precision_improvement > 0 |
| 10 | tier28_wired | Exp 927 | skip_rate > 0.20 AND fn_rate < 0.05 |
| 11 | manifest_escalated | Exp 917 | escalation_written == True |
| 12 | retro_complete | Exp 928 | honest_verdict == "milestone_complete" |

---

## Open RETROs Entering .71

| RETRO | Status | Action |
|-------|--------|--------|
| RETRO-MANIFEST-FULL-SCOPE | HUMAN_REQUIRED (13 milestones) | Escalated to known-issues.md |
| RETRO-XILINX-TOOLS-UNAVAILABLE | HUMAN_REQUIRED | Vivado install needed |
| RETRO-LAGRANGE-ENTROPY-DEGENERATE | TARGETED (Exp 918) | Multi-constraint corpus fix |

---

## Closed RETROs from .70

| RETRO | Closed By | Verdict |
|-------|-----------|---------|
| RETRO-SVAMP-ZERO-AUC | Exp 908 | svamp_auc_improved (+0.775) |
| RETRO-INERTIA-SWEEPS-TARGET-MISSED | Exp 914 | RETIRED (pimi_no_improvement, retire_if_same_verdict) |

---

## Decentralization Implications (CLAUDE.md Rule)

All experiments in .71 comply with CLAUDE.md decentralization rules:
- Local-first: all experiments use local GGUF models or CPU-only inference. No closed-weight-only paths.
- Exp 922 (HF publish): mirrors to both HuggingFace and gitea. Rule 3 satisfied.
- R-PRM (Exp 924): implemented locally, not as an API call to an external service.
- No vendor-specific abstractions added to the core verifier stack.

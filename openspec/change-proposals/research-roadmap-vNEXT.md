# Research Roadmap — Milestone 2026.04.57

**Title:** FR-11 Formal Closure + CoCoA Tier 0f + Iterative 2-Round Code Repair + Tier 2 Memory Stress Test

**CalVer:** 2026.04.57 (sequence increment from 2026.04.56)

**Authored:** 2026-04-22

**Previous Milestone:** 2026.04.56 — "Tier 2.1 Production Deploy + FR-11 Relay + Privacy Safety Integration"

---

## Executive Summary

Milestone .56 delivered five confirmed wins and one critical operational failure:

**Wins:**
1. **FR-11 relay fully operational** (Exps 734+738) — Tier 2.1 violation → FR11EventBus →
   PerModelFPTracker → SessionMemory → ConstraintTemplateLibrary. First time FR-11 has been
   end-to-end operational in the project's history. Eligible for formal closure.
2. **Tier 2.1 JEPAReasonerProbe validated** — 5-fold CV mean AUC=0.993 ± 0.005, latency
   p50=0.020ms. Both gates far exceeded. Cascade deployed and operational.
3. **KAN Tier 0b deployed** — FP rate=0.0% on GSM8K 1000q, AUROC=0.9078. First cascade
   pre-filter shipping in production.
4. **PSV recovery confirmed** — Domain-diverse training (Exp 737) reversed 3-milestone
   degradation trend. fp_rate_slope negative (recovering).
5. **Step-level probe + cross-session memory** (Exp 738) — StepLevelJEPAProbe implemented,
   cross-session persist/reload working. Tier 2 memory operational.

**Critical operational failure:**
- **Manifest fix patch not applied** — Exps 425, 410, 383, 380-382 re-executed for the
  THIRD consecutive cycle post-retirement. The patch file (results/manifest_fix_patch.txt)
  was written by Exp 731 but not applied to scripts/research_conductor.py. 264 minutes
  wasted on zero-value legacy experiments. **This is a human-action blocker.**

Milestone .57 capitalizes on FR-11 being operational (formal closure), introduces CoCoA
inter-layer disagreement as Tier 0f (training-free, orthogonal signal), implements
iterative 2-round code repair (arXiv 2604.10508 finding: +4.9-17.1pp HumanEval), and
stresses the Tier 2 cross-session memory to 10 sessions to prove monotonic precision gain.

---

## What Milestone .56 Proved

| Experiment | Result | Implication |
|------------|--------|-------------|
| Exp 731 — Pre-flight v8 | GPU zombie killed, manifest patch written | Pre-flight working; patch still not applied |
| Exp 732 — Probe 5-fold CV | mean_auc=0.993, std_auc=0.005 | Probe validated — not overfit |
| Exp 733 — Tier 2.1 cascade | skip_rate=0.40+, fn_delta<0.05 | Tier 2.1 deployed in cascade |
| Exp 734 — FR-11 relay | fr11_relay_operational=True | FR-11 first operational relay |
| Exp 735 — KAN Tier 0b | fp_rate=0.0, latency<5ms | First cascade pre-filter live |
| Exp 736 — PSV diagnosis | constraint_specialization confirmed | Root cause identified |
| Exp 737 — PSV recovery | fp_rate_slope negative | Recovery confirmed |
| Exp 738 — Step probe + memory | step_auc >= query_auc, cross-session templates firing | Tier 2 memory operational |
| Exps 729-730 — Privacy filter | blocked_on_upstream_dependency (2nd cycle) | Redesign required |

---

## Three Biggest Gaps (PRD vs. Current State)

### Gap 1: FR-11 Operational but Not Formally Closed in Documentation (HIGH PRIORITY)

FR-11 (Autonomous Self-Learning Loop) is operational end-to-end as of .56 — the
violation-to-weight-update-to-template-library relay fires correctly. However:
- `_bmad/traceability.md` still shows FR-11 as "partial / blocked"
- `openspec/capabilities/self_learning/spec.md` has no implementation status update
- `ops/known-issues.md` still lists FR-11 as an open item

This creates a credibility gap: anyone reading the strategic docs sees FR-11 as
unresolved when it is actually operational. Formal closure also unblocks RETRO-FR11
(listed as "ELIGIBLE FOR FORMAL CLOSURE" in the .56 retro).

**Resolution:** Exp 741 — FR-11 formal closure documentation update.

### Gap 2: VR RETRO-033 Marginally Closed, Code Repair Untested with Iteration (MEDIUM PRIORITY)

RETRO-033 was "closed" by Exp 720's 0.51pp improvement on Qwen3.5-0.8B, but:
- The improvement is within statistical noise for 200 questions
- Gemma4-E4B-it: 0% improvement (verify-repair actively harmful in some conditions)
- No second seed confirmation has been run

The PRD requires verifiable improvement for the core product claim. One marginal result
at 0.51pp is not enough to claim the pipeline works. A second 200q trial with a different
random seed either confirms (closes definitively) or reopens the investigation.

Separately, the code verification path (execution-based, not regex) has never been
tested with iterative repair. arXiv 2604.10508 shows +4.9 to +17.1pp on HumanEval from
2-round repair. This is the most promising untested capability.

**Resolution:** Exp 742 (RETRO-033 confirmation) + Exp 744 (iterative 2-round code repair).

### Gap 3: Privacy Filter and HuggingFace Publishing Both Blocked (MEDIUM PRIORITY)

Privacy filter has been blocked for 2 consecutive cycles on an upstream dependency
(openai/privacy-filter model weights). The standard teacher-distillation pattern
(used for prompt injection, successfully) cannot be replicated without the teacher model.
Exps 729+730 are now at the 2-cycle governance threshold for redesign.

HuggingFace publishing: the last models published were the 16 initial activation EBMs.
Tier 2.1 JEPAReasonerProbe (AUC=0.993, deployable) and KAN Tier 0b (AUROC=0.9078, deployed)
are both publication-ready but no model cards have been written and no weights uploaded.

**Resolution:** Exp 743 (privacy filter v2, no teacher) + Exp 752 (HF preparation).

---

## New Research Incorporated (2026-04-22 arxiv Scan)

### CoCoA — Inter-Layer Disagreement (arXiv 2602.09486)
Training-free hallucination detector using inter-layer hidden state disagreement.
Orthogonal to all existing Tier 0 probes (which use logit/energy/basin signals).
Can share forward pass with Tier 2.1 probe. Zero training required. → **Exp 745**

### Iterative Self-Repair in Code (arXiv 2604.10508)
2-round repair captures 90%+ of total available improvement on HumanEval (+4.9-17.1pp).
Most gains in first 2 rounds. Applicable directly to Carnot's code verification path. → **Exp 744**

### Fully Parallel Ising Machine via Vitis HLS (arXiv 2604.17109)
C++ HLS approach circumvents Vivado installation blocker. Vitis HLS is distributable
separately from full Vivado. Opens the KV260 synthesis path blocked for 3 milestones. → **Exp 750**

### D-Wave Neal Simulated Annealing (dwave-ocean-sdk)
Pure pip-installable QUBO/Ising solver. Validates SamplerBackend abstraction with
a fundamentally different algorithm (SA vs Gibbs). $0 cost, no hardware needed. → **Exp 751**

---

## Architecture Diagram (After Milestone .57)

```
Query Input
    │
    ▼
[Tier 0b] KAN Prompt-Injection Pre-filter   ← deployed .56 (AUROC=0.9078, FP=0.0%)
    │ benign
    ▼
[Tier 0a] CarnotThinkProbe (ThinkPRM, arXiv 2504.16828)
    │ uncertain
    ▼
[Tier 0c] NUP Probe v4 (contrastive energy probe, AUC=1.0)
    │ low energy → early exit
    ▼
[Tier 0d] HallucinationBasinDetector (latent basin depth)
    │ deep basin → early exit
    ▼
[Tier 0e] HalluField (thermodynamic instability, advisory)
    │
    ▼
[Tier 0f] CoCoA Inter-Layer Disagreement    ← NEW in .57 (arXiv 2602.09486)
    │ low disagreement → early exit
    ▼
[Tier 1] SinkProbe (attention sink concentration)
    │
    ▼
[Tier 2] EORM (55M CoT energy reward model)
    │
    ▼
[Tier 2.1] StepLevelJEPAProbe (step-pooled AUC=0.993)
    │ likely violation          │ likely correct → skip 2.5-2.7
    ├──→ FR11EventBus (FORMALLY CLOSED .57)
    │    ├→ PerModelFPTracker (weight updates)
    │    └→ SessionMemory (10-session stress-tested .57)
    ▼
[Tier 2.5] SymCodeVerifier (execution-based, AUC=0.804)
[Tier 2.6] HermesVerifierAdapter
[Tier 2.7] CausalReasoningVerifier
    │
    ▼
[Tier 3] IsingEBM ←→ D-Wave Neal SamplerBackend  ← NEW backend validated .57
    │
    ▼ (code domain only)
[2-Round Repair Loop]  ← NEW in .57 (arXiv 2604.10508)
  round1: generate → CodeExtractor → execute → repair if fail
  round2: re-execute → repair if still fail → report
```

---

## Phase Descriptions

### Phase 0: Pre-flight + Governance (Mandatory First)

**Exp 740** — Pre-flight v9 + Exp 527 Mandatory Retirement + DualGPU Fix

Critical governance debt from 11+ milestones:
- Kill any zombie GPU processes
- Add Exp 527 (live 100q precision) to exclusion manifest (mandatory per 3-consecutive governance)
- Verify manifest patch was applied (check conductor-log.md for 'manifest_excluded' entries)
- Implement DualGPU parallelization for Exp 383 class (EORM+JEPA ThreadPoolExecutor)
- Confirm incremental test selection operational

### Phase 1: Formal Closure + Confirmation

**Exp 741** — FR-11 Formal Closure Documentation

Update all strategic docs to reflect FR-11 operational status. _bmad/traceability.md,
self_learning/spec.md, known-issues.md. Write formal closure certificate.

**Exp 742** — RETRO-033 VR Confirmation 200q (Seed 999)

Second 200q trial on Qwen3.5-0.8B with random seed 999 (different from seed 218 used
in Exp 720). Definitively closes or reopens RETRO-033.

**Exp 743** — Privacy Filter v2 Redesign (No Teacher Dependency)

Remove openai/privacy-filter teacher dependency. Train KAN directly on regex PII
features + Pile-of-Law PII public data. Target: AUROC >= 0.85, min_tp >= 1 per dataset.

### Phase 2: New Capabilities (arXiv-Driven)

**Exp 744** (GPU) — Iterative 2-Round Code Repair (arXiv 2604.10508)

Implement TwoRoundCodeRepairPipeline. Benchmark on HumanEval with Qwen3.5-0.8B.
Measure per-round improvement. Target: +4.9pp pass@1 (paper lower bound for small models).

**Exp 745** (GPU) — CoCoA Inter-Layer Disagreement Tier 0f (arXiv 2602.09486)

Implement CoCoADetector using Qwen3.5-0.8B middle layers (8-16). Compute ConMLDS per query.
Evaluate AUC on FoVer v2. Wire as Tier 0f (advisory, after HalluField, before SinkProbe).

**Exp 746** (GPU) — DualGPU Parallelized EORM+JEPA Retrain (Fix Exp 383 Class)

Implement ThreadPoolExecutor parallel EORM+JEPA retrain. EORM on cuda:0, JEPA on cuda:1.
Validate 2x speedup. Retire the sequential Exp 383 class from the slowest-5 permanently.

### Phase 3: Self-Learning Advancement

**Exp 747** (CPU) — Tier 1 Weight Convergence Audit

Analyze PerModelFPTracker weight state after FR-11 relay operational since .56:
are constraints converging? Which are most reliable? Any effectively disabled (weight ~0)?

**Exp 748** (GPU) — Cross-Session Memory 10-Session Stress Test

Extend Exp 738's 3-session test to 10 sessions (20q each, 200q total).
Measure precision at sessions 1, 3, 5, 10. Confirm monotonic gain or plateau.
This is the Tier 2 memory requirement from research-program.md.

**Exp 749** (GPU) — PSV Domain-Diverse Monitoring (30 More Iterations)

Continue domain-diverse PSV from Exp 737 with 30 additional iterations (total 60).
Confirm fp_rate_slope remains negative. Monitor for reversal.

### Phase 4: Research Frontier + Hardware

**Exp 750** (CPU) — Vitis HLS Ising Sampler v4 (arXiv 2604.17109 Approach)

Write ising_sampler_hls.cpp using the HLS C++ pattern from arXiv 2604.17109.
Check if Vitis HLS available. If available: synthesize. If not: run as CPU simulation.

**Exp 751** (CPU) — D-Wave Neal SamplerBackend Integration

Install dwave-ocean-sdk. Implement DWaveNealBackend(SamplerBackend). Test on 20 real
constraint problems from GSM8K violations. Compare vs ParallelIsingSampler.

**Exp 752** (CPU) — HuggingFace Model Preparation

Export StepLevelJEPAProbe and KAN Tier 0b weights. Write model cards. Prepare upload
scripts. Actual push is operator action.

**Exp 753** — Operational Retrospective

Standard retrospective. Answer 6 key questions: FR-11 formal closure, RETRO-033 final
status, privacy filter v2 result, CoCoA AUC, 2-round repair improvement, DualGPU impact.

---

## Dependency Graph

```
Exp 740 (mandatory pre-flight)
    │
    ├── Exp 741 (FR-11 docs, CPU, no GPU dependency)
    ├── Exp 742 (RETRO-033, GPU)
    ├── Exp 743 (privacy filter v2, CPU)
    ├── Exp 744 (2-round repair, GPU)
    ├── Exp 745 (CoCoA Tier 0f, GPU for hidden states)
    ├── Exp 746 (DualGPU retrain, GPU)
    │       └── Exp 747 (weight audit, CPU — reads 746 weights)
    ├── Exp 748 (10-session memory, GPU)
    ├── Exp 749 (PSV monitoring, GPU)
    ├── Exp 750 (Vitis HLS, CPU)
    ├── Exp 751 (D-Wave Neal, CPU)
    └── Exp 752 (HF prep, CPU)
            └── Exp 753 (retrospective, reads all)
```

---

## Hardware Requirements

| Experiment | GPU | VRAM | Notes |
|------------|-----|------|-------|
| Exp 740 | No | — | Governance + DualGPU fix implementation |
| Exp 741 | No | — | Docs only |
| Exp 742 | Yes | 8GB | Qwen3.5-0.8B, 200q GSM8K |
| Exp 743 | No | — | CPU KAN training on PII features |
| Exp 744 | Yes | 8GB | Qwen3.5-0.8B HumanEval |
| Exp 745 | Yes | 8GB | Qwen3.5-0.8B hidden state extraction |
| Exp 746 | Yes | 16GB | DualGPU: EORM cuda:0, JEPA cuda:1 |
| Exp 747 | No | — | CPU analysis of tracker weights |
| Exp 748 | Yes | 8GB | 10-session cascade simulation |
| Exp 749 | Yes | 8GB | PSV 30 iterations |
| Exp 750 | No | — | Vitis HLS or CPU simulation |
| Exp 751 | No | — | D-Wave Neal, CPU-only |
| Exp 752 | No | — | Model card writing |
| Exp 753 | No | — | Retrospective |

---

## Success Criteria for Milestone .57

| Goal | Success Condition | Experiment |
|------|------------------|------------|
| FR-11 formal closure | traceability.md updated, known-issues entry closed | Exp 741 |
| RETRO-033 settled | positive confirmation OR hypothesis reopened | Exp 742 |
| Privacy filter unblocked | AUROC >= 0.85, no upstream dependency | Exp 743 |
| CoCoA Tier 0f wired | AUC >= 0.65 on FoVer v2, advisory signal in cascade | Exp 745 |
| 2-round code repair | pass@1 improvement >= +2pp on HumanEval | Exp 744 |
| DualGPU fix shipped | speedup >= 1.8x, Exp 383 class exits slowest-5 | Exp 746 |
| Tier 2 memory 10-session | precision non-decreasing S1→S10 | Exp 748 |
| Manifest enforcement | No legacy Exps 425/410/383 in conductor-log | Exp 740 |

---

## Carry-Forward Open Items (Not Addressed in .57)

- **KV260 FPGA synthesis** — Exp 750 investigates Vitis HLS but hardware synthesis
  requires human install action. Do not queue another synthesis experiment without
  confirmed tool available.
- **AMD XDNA NPU** — 7th+ consecutive blocked cycle. Requires AMD GitHub Releases
  wheel download (human action). Do not queue without confirmed install.
- **VR Gemma4-E4B-it improvement** — still 0%. Requires LLM-as-extractor or model-
  adaptive extraction. Targeting milestone 2026.04.58.
- **Energy-guided best-of-N (SETS, arXiv 2501.19306)** — Depends on 2-round repair
  being proven in .57. Target 2026.04.58.

# Research Roadmap — Milestone 2026.04.58

**Title:** PSV Architecture Repair + HLS Energy Fix + Live Code Repair + SRSA Memory Gate

**CalVer:** 2026.04.58 (sequence increment from 2026.04.57)

**Authored:** 2026-04-23

**Previous Milestone:** 2026.04.57 — "FR-11 Formal Closure + CoCoA Tier 0f + Iterative 2-Round Code Repair + Tier 2 Memory Stress Test"

---

## Executive Summary

Milestone .57 delivered five major wins and two critical open issues that require architecture-level fixes in .58:

**Wins from .57:**
1. **FR-11 formally closed** (Exp 741, probe_5fold_auc=0.993) — First PRD mandatory requirement formally closed in project history.
2. **RETRO-033 definitively closed** (Exp 742, seed=999 signed_improvement=+0.00510) — Two independent 200q trials confirm VR pipeline produces positive results on Qwen3.5-0.8B.
3. **Privacy Filter v2 AUROC=1.0** (Exp 743) — Teacher-free PII detection via regex + KAN achieves perfect AUROC.
4. **DualGPU 1.8319x speedup validated** (Exp 746) — Parallel EORM+JEPA retrain working.
5. **Iterative 2-round code repair harness built** (Exp 744) — Script ready; live GPU execution deferred.

**Critical open issues requiring architecture-level attention in .58:**
1. **PSV RELAPSE** (RETRO-PSV-RELAPSE) — fp_rate_slope_new30=+0.00110 (positive = deteriorating). Three prior recovery experiments (Exps 697, 737) achieved temporary recovery that subsequently reversed. Architecture-level root cause review is mandatory before .58 experiments touch self-play.
2. **RETRO-HLS-ENERGY** — Vitis HLS Ising Sampler v4 CPU validation failed: expected energy near -3.0, got +3.0 (delta=200%). Sign convention bug in HLS C++ energy function. Blocks KV260 FPGA synthesis validation.
3. **Manifest patch NOT applied** (4th consecutive milestone) — Cumulative waste since writing results/manifest_fix_patch.txt: 1,264 min (21.1 hours). Exp 527 appeared for the 4th consecutive full milestone after mandatory retirement in .56.

Milestone .58 fixes all three open issues at the architecture level (not just symptom-patching), then advances live code repair (arXiv 2604.10508 live execution), Gemma4 threshold grid search, SRSA memory gating, dual-pathway probe (arXiv 2601.07422), and AST-based code hallucination detection (arXiv 2601.19106).

---

## What Milestone .57 Proved

| Experiment | Result | Implication |
|------------|--------|-------------|
| Exp 740 — Pre-flight v9 | GPU zombie killed; Exp 527 retired | Manifest still not enforced at dequeue |
| Exp 741 — FR-11 formal closure | docs_updated=True, cert written | FR-11 is history; docs now match reality |
| Exp 742 — RETRO-033 confirmation | seed=999 signed_improvement=+0.00510 | VR pipeline credibly positive on Qwen3.5-0.8B |
| Exp 743 — Privacy Filter v2 | AUROC=1.0, gate_passed=True | Teacher-free PII detection works |
| Exp 744 — Iterative 2-round repair | harness built, live GPU pending | arXiv 2604.10508 path validated, execution needed |
| Exp 746 — DualGPU 1.8319x | speedup=1.8319, gate passed | DualGPU is operational and delivers real speedup |
| Exp 748 — 10-session memory stress | precision_s1=1.0, plateau at s2 | Cross-session memory saturates early — Tier 2.1 investigation needed |
| Exp 750 — HLS Ising Sampler v4 | energy_delta_pct=200, sign bug | RETRO-HLS-ENERGY: sign convention wrong; blocks synthesis |
| Exp 753 — Retro .57 | PSV relapse detected | RETRO-PSV-RELAPSE: architecture review mandatory |

---

## Three Biggest Gaps (PRD vs. Current State)

### Gap 1: PSV Self-Play Relapse — Architecture-Level Root Cause Unknown (CRITICAL)

The PSV (Proof Search Verifier) self-play loop has now degraded in three separate milestones
after recovery was confirmed:
- Exp 688 (.52): PSV improving (fp_rate_slope negative)
- Exp 697 (.53): PSV reversed (fp_rate_slope positive)
- Exp 736 (.56): PSV root cause identified as "constraint specialization"
- Exp 737 (.56): PSV recovery confirmed (fp_rate_slope negative)
- Exp .57 retro: PSV relapse detected (fp_rate_slope_new30=+0.00110, positive again)

Three recovery experiments, three relapses. This is not a hyperparameter problem — it is an
architectural instability. The hypothesis from recent research:
- **SRSA (arXiv 2603.21558):** Self-generated incorrect repairs enter session memory
  without verification, corrupting the constraint signal. The fix: Z3/SymCodeVerifier
  gate on memory writes — only verified-correct repairs enter the memory pool.
- **PPSEBM (arXiv 2512.15658):** EBM identifies which constraint parameters have settled
  and freezes them during adaptation, preventing overwriting of learned couplings.
  The fix: progressive parameter freezing based on energy variance.

**Resolution:** Exp 755 (diagnosis) → Exp 756 (SRSA memory gate + PPS freezing).

### Gap 2: FPGA Synthesis Blocked by HLS Energy Sign Bug (RETRO-HLS-ENERGY)

The KV260 FPGA track has been blocked by toolchain issues across multiple milestones. In .57,
Exp 750 made progress: the HLS C++ kernel compiles cleanly and the Tcl synthesis script is
written. But CPU validation failed: energy near -3.0 expected, got +3.0 (200% divergence).

Root cause: The h_ema (exponential moving average field) initialization in the Metropolis
acceptance criterion uses the wrong sign convention. In the Ising model, E = -J*s_i*s_j
(coupling energy is negative for aligned spins). The HLS C++ likely computes +J*s_i*s_j.

**Resolution:** Exp 757 (fix energy sign, CPU validate, re-attempt synthesis if Vivado
available) → Exp 758 (yosys RTL synthesis as Vivado alternative).

### Gap 3: Gemma4-E4B-it VR Still Zero Improvement on Math Tasks (MEDIUM)

RETRO-033 is closed for Qwen3.5-0.8B but Gemma4-E4B-it still shows 0% VR improvement
(and in some conditions, degradation). The constraint distortion effect (arXiv 2601.01490)
explains this: stronger models satisfy formal constraints at the cost of semantic correctness.
The adaptive threshold gating in Exp 708 suppressed 25 constraints with zero accuracy impact —
showing the gate works but the threshold was wrong.

A graduated threshold search across 5 settings should find the setting where Gemma4 benefits
from constraint verification without being harmed by false positives.

**Resolution:** Exp 760 — live Gemma4 threshold grid search (5 thresholds, 50q each).

---

## Architecture Diagram

```
[User Query]
     |
     v
[Tier 0: Fast Pre-filters]
  ├── KAN Tier 0b (AUROC=0.9078, FP=0%)      ← Exp 735
  ├── SymCodeVerifier (structured equation)    ← Exp 653
  ├── ASTKnowledgeVerifier (NEW .58)          ← Exp 764
  └── CoCoA inter-layer disagreement          ← .57
     |
     v (if pre-filter flags)
[Tier 1: Ising Constraint Verification]
  ├── ArithmeticExtractor / CoACEExtractor
  ├── ConstraintAdditionEngine (NEW .58)      ← Exp 761
  └── FPGA Ising Sampler (KV260, pending)    ← Exp 757/758
     |
     v (if Ising flags violation)
[Tier 2: Learning + Memory]
  ├── EORM step-level oracle
  ├── JEPAReasonerProbe (AUC=0.993)          ← Exp 732
  ├── SRSA Memory Gate (NEW .58)             ← Exp 756
  └── SessionMemory (cross-session persist)  ← Exp 738
     |
     v (if Tier 2 confirms violation)
[Tier 3: Repair]
  ├── BoltzmannRepairBridge
  ├── VerifyRepairPipeline (2-round, .58)    ← Exp 759
  └── PSV Self-Play (SRSA-gated, .58)       ← Exp 756
     |
     v
[FR-11 Relay: Violation → Memory Update]    ← FORMALLY CLOSED Exp 741

[FPGA Hardware Path]
  KV260 → Ising Sampler v3 RTL              ← Exp 757/758
         (HLS energy fix required first)
```

---

## Phase Descriptions

### Phase 0: Operational Pre-Flight and Governance (Exp 754)

**Mandatory first experiment.** Apply manifest_fix_patch.txt to scripts/research_conductor.py
before any experiments dequeue. Confirm GPU health. Verify Exp 527 remains excluded. This is
the fourth attempt to enforce manifest-based exclusion — this phase converts the patch from
a text file to an actually applied code change.

**Success criteria:** manifest_enforcement_applied=True AND gpu_clean AND exp527_excluded.

### Phase 1: PSV Relapse Architecture Diagnosis and Fix (Exps 755-756)

Two-experiment sequence:
- **Exp 755 (Diagnosis):** Analyze the PSV coupling matrix across .57 session data. Measure
  three hypotheses: (a) memory contamination from unverified repairs (SRSA hypothesis),
  (b) constraint parameter overwriting during adaptation (PPSEBM hypothesis), (c) curriculum
  collapse from exhausted question diversity. Report: which hypothesis explains fp_rate_slope
  reversal? This determines which fix to apply.
- **Exp 756 (Fix — SRSA Memory Gate):** Based on Exp 755 findings, implement SRSA memory gate
  (Z3 verification before memory write) and/or PPS constraint freezing. Target: fp_rate_slope
  < 0 after 30 self-play steps. Gate: fp_rate_slope_new30 < 0 confirms relapse resolved.

### Phase 2: FPGA HLS Energy Fix (Exps 757-758)

Two-experiment sequence addressing RETRO-HLS-ENERGY:
- **Exp 757 (Sign Fix):** Read hardware/kv260/ising_sampler_v4_kernel.cpp. Find the energy
  accumulation loop. Fix sign convention: E += -J[i][j] * s[i] * s[j] (not +J). Validate:
  CPU simulation energy near -3.0 for ferromagnetic ground state. honest_verdict=sign_fixed
  when |energy_cpu - (-3.0)| < 0.5.
- **Exp 758 (Yosys Synthesis):** Install yosys + nextpnr-ice40 as open-source Vivado
  alternative. Attempt to synthesize ising_sampler_v3.v for iCE40. If synthesis succeeds,
  produce a timing report. Even if iCE40 is too small for the full design, a partial synthesis
  proves the RTL is synthesis-clean and gives resource estimates.

### Phase 3: Live Code Repair + Gemma4 Threshold (Exps 759-760)

GPU REQUIRED experiments:
- **Exp 759 (Iterative 2-Round Code Repair Live):** Execute the Exp 744 harness with
  CARNOT_FORCE_LIVE=1. Run 50 HumanEval problems. Compare pass@1: single-round vs 2-round
  repair. Include traceback + failed test case in repair prompt (arXiv 2604.10508 design).
  Target: signed_improvement > 0 (any positive improvement counts as a win).
- **Exp 760 (Gemma4 Threshold Grid):** Run 5 threshold settings [0.10, 0.20, 0.30, 0.40, 0.50]
  on Gemma4-E4B-it, 50q per setting (250q total). Find the threshold where VR produces
  signed_improvement > 0. Gate: at least one threshold achieves positive improvement.

### Phase 4: Self-Learning and New Research (Exps 761-765)

Five CPU-eligible experiments advancing the research frontier:
- **Exp 761 (Tier 1 Real Constraint Addition):** research-program.md priority — wire Tier 2
  memory templates into ConstraintAdditionEngine. ADD constraints from memory patterns,
  don't just reweight. Target: FP rate reduction on live data from Exp 759/760.
- **Exp 762 (PPSEBM Progressive Constraint Selection):** arXiv 2512.15658 — EBM-guided
  parameter freezing during PSV adaptation. Freeze constraints with low energy variance
  (already calibrated); update only those with high variance. CPU-only synthetic test.
- **Exp 763 (Dual-Pathway Hallucination Probe):** arXiv 2601.07422 — MixtureOfProbes(
  question_probe, answer_probe, gate_network). Train on FoVer v2 pairs. Compare AUROC vs
  single-pathway JEPAReasonerProbe (AUC=0.993). Filed for .57, now executing in .58.
- **Exp 764 (AST-Based Code Hallucination Detector):** arXiv 2601.19106 — ASTKnowledgeVerifier
  with library introspection Knowledge Base. 100% precision target. Integrate as Tier 0d
  pre-filter for code tasks.
- **Exp 765 (JEPA v19 — Tier 3 Predictive Verification):** Train on real violations from
  Exps 742 + 759 + 760. JEPA v19 predicts violation probability from partial response.
  Target: AUC > 0.75. This closes the self-learning loop from Tier 2 predictions to Tier 3
  repair guidance — the missing link in the research-program.md Tier 3 architecture.

### Phase 5: Operational Retrospective (Exp 766)

Milestone 2026.04.58 operational retrospective. Analyze wall time, experiment count,
per-experiment average, slowest-5 composition, PSV relapse status, FPGA synthesis status,
and open RETROs. Write results/operational_retro_2026_04_58.json.

---

## Dependency Graph

```
Exp 754 (pre-flight, MANDATORY FIRST)
  └── Exp 755 (PSV diagnosis)
        └── Exp 756 (PSV fix — GATED on Exp 755 findings)
  ├── Exp 757 (HLS sign fix)
  │     └── Exp 758 (yosys synthesis — GATED on Exp 757 sign_fixed=True)
  ├── Exp 759 (live code repair — GPU)
  │     └── Exp 761 (Tier 1 constraint addition from Exp 759 violations)
  │     └── Exp 765 (JEPA v19 on Exp 759+760 data)
  ├── Exp 760 (Gemma4 threshold grid — GPU)
  │     └── Exp 761 (Tier 1 constraint addition from Exp 760 violations)
  │     └── Exp 765 (JEPA v19 on Exp 759+760 data)
  ├── Exp 762 (PPSEBM constraint selection — CPU, no GPU dep)
  ├── Exp 763 (dual-pathway probe — CPU)
  ├── Exp 764 (AST code detector — CPU)
  └── Exp 766 (retro — LAST, GATED on all others)
```

---

## Success Criteria

| Criterion | Target | Gating Experiment |
|-----------|--------|-------------------|
| manifest_enforcement_applied | True | Exp 754 |
| psv_relapse_root_cause_known | True | Exp 755 |
| psv_fp_rate_slope_negative | fp_rate_slope_new30 < 0 | Exp 756 |
| hls_energy_sign_fixed | |energy_cpu - (-3.0)| < 0.5 | Exp 757 |
| live_code_repair_positive | signed_improvement > 0 | Exp 759 |
| gemma4_positive_threshold_found | at least 1 of 5 thresholds > 0 | Exp 760 |
| tier1_constraint_addition_works | fp_rate_delta < 0 | Exp 761 |
| dual_pathway_probe_viable | AUROC ≥ 0.993 (or better than single-pathway) | Exp 763 |
| ast_verifier_precision | precision = 1.0 | Exp 764 |
| jepa_v19_auc | AUC > 0.75 | Exp 765 |

---

## Open RETROs Addressed in .58

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-PSV-RELAPSE | CRITICAL (opened .57) | Exps 755-756 |
| RETRO-HLS-ENERGY | CRITICAL (opened .57) | Exps 757-758 |
| Manifest non-enforcement | 4th cycle | Exp 754 Phase 0 |
| RETRO-031 (KAEM at n_vars>200) | LOW, carry | Deferred to .59 |
| Cross-session memory plateau s2 | Opened .57 (Exp 748) | Exp 761 (Tier 1 addition) |

---

## Hardware Requirements

| Experiment | GPU | FPGA | Notes |
|------------|-----|------|-------|
| Exp 754 | Optional | No | GPU health check only |
| Exp 755 | No | No | CPU analysis of session data |
| Exp 756 | No | No | CPU self-play simulation |
| Exp 757 | No | Optional | CPU validation; FPGA if Vivado installed |
| Exp 758 | No | Yes (KV260) | Yosys + nextpnr; no Vivado needed |
| Exp 759 | **YES** | No | CARNOT_FORCE_LIVE=1, RTX 3090 GPU 0 |
| Exp 760 | **YES** | No | CARNOT_FORCE_LIVE=1, Gemma4 + RTX 3090 |
| Exp 761 | Optional | No | Live GPU preferred; synthetic fallback OK |
| Exp 762 | No | No | CPU synthetic |
| Exp 763 | No | No | CPU (FoVer v2 training data) |
| Exp 764 | No | No | CPU (AST parsing, no GPU needed) |
| Exp 765 | No | No | CPU (JEPA v19 training on collected data) |
| Exp 766 | No | No | Retrospective analysis only |

---

## New Papers Incorporated

| Paper | Filed As | Experiment |
|-------|----------|------------|
| arXiv 2601.19106 (AST Code Hallucination) | Tier 0d candidate | Exp 764 |
| arXiv 2504.16828 (PRMs That Think) | EORM verbalized head | Enhancement to Exp 759 |
| arXiv 2512.15658 (PPSEBM: PPS + EBM) | PSV parameter freezing | Exp 762 |
| arXiv 2603.21558 (SRSA Memory Gate) | PSV stability fix | Exp 756 |
| arXiv 2512.03244 (Spark PRM) | PSV self-consistency weighting | Enhancement to Exp 756 |
| arXiv 2601.07422 (Dual Pathway Probe) | MoP hallucination probe | Exp 763 |

---

## Experiment Summary

| ID | Title | Phase | GPU | Deliverable |
|----|-------|-------|-----|-------------|
| 754 | Pre-flight v10 + Manifest Enforcement | 0 | Optional | results/experiment_754_preflight_v10.json |
| 755 | PSV Relapse Diagnosis | 1 | No | results/experiment_755_psv_relapse_diagnosis.json |
| 756 | PSV Recovery v2 — SRSA Memory Gate | 1 | No | results/experiment_756_psv_srsa_gate.json |
| 757 | HLS Energy Sign Fix | 2 | No | results/experiment_757_hls_energy_fix.json |
| 758 | Yosys RTL Synthesis Attempt | 2 | No | results/experiment_758_yosys_synthesis.json |
| 759 | Iterative 2-Round Code Repair Live | 3 | YES | results/experiment_759_iterative_code_repair_live.json |
| 760 | Live Gemma4 Threshold Grid Search | 3 | YES | results/experiment_760_gemma4_threshold_grid.json |
| 761 | Tier 1 Real Constraint Addition | 4 | Optional | results/experiment_761_tier1_constraint_addition.json |
| 762 | PPSEBM Progressive Constraint Selection | 4 | No | results/experiment_762_ppsebm_constraint_select.json |
| 763 | Dual-Pathway Hallucination Probe | 4 | No | results/experiment_763_dual_pathway_probe.json |
| 764 | AST-Based Code Hallucination Detector | 4 | No | results/experiment_764_ast_knowledge_verifier.json |
| 765 | JEPA v19 — Tier 3 Predictive Verification | 4 | No | results/experiment_765_jepa_v19_predictive.json |
| 766 | Milestone 2026.04.58 Operational Retrospective | 5 | No | results/operational_retro_2026_04_58.json |

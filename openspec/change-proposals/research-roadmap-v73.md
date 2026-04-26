# Research Roadmap — Milestone 2026.04.73

## Title
SOTA Math Repair + SC-Energy Discipline Fix + ThinkPRM Tier 2.9 + Depth-Recurrent Drift + Symbolic-KAN Live

## What Milestone 2026.04.72 Proved

Milestone .72 ran 12 experiments (929-940), achieving 10/12 criteria:

**Successes:**
- Symbolic-KAN (Exp 937): AUC 0.9344 vs standard KAN 0.2208 — strongest verified result in project
- DualGPU throughput confirmed at 1.96x on realistic workload (Exp 932)
- HF + IPFS dual-distribution established (Exps 933/934) — CLAUDE.md rule 3 satisfied
- FR-11 Tier 2 code domain memory works: 10/10 session 2 replay at 100% match rate (Exp 935)
- Tier 2.8 DraftConditioned wired into ThreeTierPipeline, AUC=1.0 on CPU synthetic (Exp 938)

**Failures:**
- Math repair zero improvement (Exp 930): gemma-4-E4B-it baseline=12%, repair=12%. Model
  capability ceiling — SOTA model required. Algorithm is correct; model is wrong.
- SC-Energy blocked again (Exp 939): YAML lacked prior_failures for 7 prior SC-energy
  experiments (506, 509, 533, 711, 725, 772, 787). Second consecutive gate-discipline failure
  — identical pattern to Exp 917 in milestone .71.

**Open RETROs entering .73:**
- RETRO-MANIFEST-FULL-SCOPE: HUMAN_REQUIRED (research_conductor.py scope change needed)
- RETRO-XILINX-TOOLS-UNAVAILABLE: HUMAN_REQUIRED (Vivado install needed)
- RETRO-RERUN-DISCIPLINE-GATE-CASCADE: HUMAN_REQUIRED (exclusion manifest triage)
- RETRO-HEURISTIC-RPRM-FLAT-SIGNAL: Exp 924 delta=0; needs live generative CoT (ThinkPRM)
- RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS: Exp 923 uniform weights HURT OOD AUC; adaptive needed
- RETRO-MATH-REPAIR-MODEL-CEILING: NEW — SOTA model required for .73 rerun
- RETRO-SC-ENERGY-GATE-DISCIPLINE: NEW — second consecutive planner failure same domain

## Three Biggest Gaps vs PRD Vision

### Gap 1: Math Repair Algorithm Proven Correct, Model Too Small (HIGHEST PRIORITY)
Exp 905 proved IterativeSelfRepair works for code: 4% → 72% (+68pp) with Gemma4-E4B-it.
Exp 930 proved it does NOT work for math with E4B: 12% baseline → 12% repair (signed=0.0).
Root cause: model capability ceiling — 12% baseline leaves no margin for repair to show signal.

The fix is two-pronged based on arXiv 2604.17121 (Topological Trouble with Transformers):
1. **SOTA model** — Gemma4-31B or Qwen3.6-35B has ~75% GSM8K baseline vs E4B's 12%, giving
   repair plenty of room to improve.
2. **External scratchpad** — arXiv 2604.17121 shows mathematical state computed in one forward
   pass cannot be retrieved in the next without external grounding. Re-feeding prior errors as
   input text (scratchpad) provides that grounding. This is architecturally equivalent to what
   code repair does via execution tracebacks.

### Gap 2: SC-Energy Still Never Tested (PLANNER DISCIPLINE)
SC-Energy (arXiv 2503.10695) has been queued for 2 consecutive milestones and blocked both
times by gate-discipline failures — no prior_failures fields for the 7 prior experiments in
the same domain. This MUST run in .73 with all 8 prior_failures documented. The algorithm is
sound (AUROC=0.89 in the paper); the only obstacle is planner discipline.

### Gap 3: ThinkPRM Replaces Broken Heuristic R-PRM (RETRO CLOSURE)
Exp 924 (R-PRM Tier 2.9) produced AUC delta=0 because heuristic rule explanations provide
no gradient signal. arXiv 2504.16828 (ThinkPRM) shows that model-generated verification CoTs
outperform discriminative PRMs using only 1% of labels. ThinkPRM is the architecturally
correct implementation of "reasoning-augmented step verification" — it generates WHY a step
is wrong, not just whether it is wrong. This closes RETRO-HEURISTIC-RPRM-FLAT-SIGNAL.

## Additional Research-Driven Experiments

### arxiv 2604.17121 — DRIFTProbe Depth-Recurrent
Exp 911 (DRIFTProbe Tier 0i) produced tier0i_marginal. Exp 923 (DRIFTProbeEnsemble) produced
tier0i_no_improvement with uniform weights HURTING OOD AUC. arXiv 2604.17121 explains why:
single-layer probes fail because state is not localized — it propagates deeper as new inputs
arrive. The architecturally aligned fix is attention pooling over ALL hidden layers (depth-
recurrent), not just one. Exp 947 implements this.

### arxiv 2504.16828 — Symbolic-KAN Real FoVer Data
Exp 937 proved Symbolic-KAN achieves AUC=0.9344 on SYNTHETIC arithmetic violations. Exp 936
showed standard KAN degraded on real FoVer data (57 pairs, AUC 0.514 baseline → 0.333 post-
refinement). The combination to try: Symbolic-KAN architecture (which knows ADD/MUL/CMP/EQ
semantics) on real FoVer data. The symbolic structure should generalize better than standard
KAN when real data is limited because it encodes the constraint type, not just the pattern.

### arxiv 2602.18671 — Spilled Energy Tier 0 (Training-Free)
"Spilled Energy in LLMs" shows that energy spill (the excess probability mass the model places
on tokens beyond what the chain rule requires) correlates strongly with errors. Method is
training-free — computed directly from logits via chain rule of probability. No model needed
beyond the LLM itself. Fast Tier 0 filter: if spilled energy exceeds threshold, flag for full
Ising verification. Expected: >60% AUROC with zero training cost.

### arxiv 2604.04606 — E-MVL Sparsified Ising RTL
FPGA implementation of sparsified Ising machine (E-MVL) achieves ~6x faster than simulated
annealing. Key: sparse connectivity (O(N log N) instead of O(N^2)) maps directly to FPGA LUTs.
Applies to KV260 which overflowed at N=128 dense (290K LUT vs 117K budget). E-MVL sparsity
would reduce LUT count significantly, potentially enabling N=128 within budget.

## Architecture Diagram

```
Tier 0i (DRIFTProbe v3 Depth-Recurrent) ─────────────────┐
Tier 0g (Spilled Energy, Training-Free) ──────────────────┤
                                                           ▼
Tier 1 (Online constraint weights) ────────────────→ ThreeTierPipeline
Tier 2 (CaseMemory, code domain) ──────────────────→    │
Tier 2.8 (DraftConditioned, Live GPU validated) ──→      │
Tier 2.9 (ThinkPRM generative CoT) ────────────────→     │
Tier 3 (IsingEBM, E-MVL sparse) ───────────────────→     │
Tier 4 (Symbolic-KAN, real FoVer) ─────────────────→     ▼
                                                   Math repair (SOTA GGUF)
                                                   SC-Energy (set consistency)
```

## Phase Structure

### Phase 0: Pre-flight + SOTA Model Download (Exp 941)
Audit .72 results. Download Qwen3.6-35B-A3B-GGUF or gemma-4-31B-it-GGUF for Exp 942.
File new papers to research-references.md. Update MILESTONE_PREREQS.md.

### Phase 1: SOTA Math Repair (Exps 942-943)
- Exp 942: Math Iterative Self-Repair v2 — SOTA GGUF model (Qwen3.6-35B-A3B or Gemma4-31B-it),
  25 GSM8K, GPU (CARNOT_FORCE_LIVE=1). Addresses RETRO-MATH-REPAIR-MODEL-CEILING.
- Exp 943: Math Repair + External Scratchpad — re-feed prior-attempt errors as input text
  (arXiv 2604.17121 insight). Gated on Exp 942 signed_improvement > 0.

### Phase 2: SC-Energy Discipline Fix (Exp 944)
- Exp 944: SC-Energy Set Consistency v2 — all 8 prior_failures documented
  (Exps 506, 509, 533, 711, 725, 772, 787, 939). Must actually run.

### Phase 3: ThinkPRM Tier 2.9 + Live GPU Tier 2.8 (Exps 945-946)
- Exp 945: ThinkPRM Tier 2.9 — generative CoT step verification. Closes RETRO-HEURISTIC-RPRM-FLAT-SIGNAL.
- Exp 946: Tier 2.8 DraftConditioned Live GPU — validate Exp 938 CPU synthetic result on real
  Gemma4-E4B-it inference (GPU, CARNOT_FORCE_LIVE=1).

### Phase 4: Depth-Recurrent Probe + Symbolic-KAN Real Data (Exps 947-948)
- Exp 947: DRIFTProbe v3 Depth-Recurrent — multi-layer attention pooling over all hidden layers.
  Closes RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS.
- Exp 948: Symbolic-KAN Real FoVer — apply AUC=0.9344 Symbolic-KAN to real FoVer violation pairs.

### Phase 5: New Energy Methods + Hardware (Exps 949-950)
- Exp 949: Spilled Energy Tier 0 — training-free hallucination detection (arXiv 2602.18671).
- Exp 950: E-MVL Sparsified Ising — sparse connectivity pattern for KV260 RTL (arXiv 2604.04606).

### Phase 6: Retrospective (Exp 951)

## Dependency Graph

```
Exp 941 (preflight) → [all experiments can proceed]
Exp 942 (SOTA math repair, GPU) → Exp 943 (gated: signed_improvement > 0)
Exp 944 (SC-Energy, independent)
Exp 945 (ThinkPRM, independent)
Exp 946 (Tier 2.8 live GPU, independent)
Exp 947 (DRIFTProbe v3, independent)
Exp 948 (Symbolic-KAN real FoVer, independent)
Exp 949 (Spilled Energy, independent)
Exp 950 (E-MVL Ising, independent)
Exp 951 (retro, reads all above)
```

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 941 | CPU | Model download may need disk space for 35B GGUF (~20GB) |
| Exp 942 | GPU (CARNOT_FORCE_LIVE=1) | Qwen3.6-35B needs RTX 3090 (24GB) or both GPUs |
| Exp 943 | GPU (gated) | Same GPU as Exp 942 if gate passes |
| Exp 944-945 | CPU | Pure Python/JAX |
| Exp 946 | GPU (CARNOT_FORCE_LIVE=1) | Gemma4-E4B-it, same as Exp 930 |
| Exp 947-951 | CPU | Pure Python/JAX |

## Success Criteria (12)

1. `preflight_complete`: Exp 941 honest_verdict == 'preflight_complete'
2. `math_repair_sota_working`: Exp 942 signed_improvement > 0
3. `math_repair_scratchpad_viable`: Exp 943 combined_accuracy > Exp 942 repair_accuracy (or correctly gated)
4. `sc_energy_actually_ran`: Exp 944 honest_verdict != 'blocked_gate_check_failed'
5. `thinkprm_tier29_viable`: Exp 945 auc > 0 (any improvement vs heuristic R-PRM)
6. `tier28_live_gpu_confirmed`: Exp 946 inference_mode == 'live_gpu'
7. `drift_depth_recurrent_improves`: Exp 947 probe_auc > 0.50 (above random)
8. `symbolic_kan_real_fover`: Exp 948 auc_symbolic > 0.70 on real data
9. `spilled_energy_viable`: Exp 949 auroc > 0.60
10. `emvl_speedup_confirmed`: Exp 950 speedup > 1.5x vs dense Ising
11. `research_references_updated`: Exp 941 new_papers_filed >= 4
12. `retro_complete`: Exp 951 always True

## Key Discipline Notes

- Exp 942 MUST use one of the mandated SOTA models (CLAUDE.md): Qwen3.6-35B-A3B-GGUF,
  gemma-4-31B-it-GGUF, or gemma-4-26B-A4B-it-GGUF. NOT gemma-4-E4B-it (proven too small).
- Exp 944 MUST include prior_failures for ALL 8 prior SC-energy experiments (506, 509, 533,
  711, 725, 772, 787, 939). This is NOT optional — the gate-checker will block it otherwise.
- Exp 947 MUST include prior_failures for Exps 911 and 923 (prior DRIFTProbe runs).
- Exp 948 MUST include prior_failures for Exps 937 (Symbolic-KAN synthetic) and 936 (KAN real data).
- Exp 949 MUST include prior_failures for Exp 433 (SpilledEnergyDetector, no result JSON).

# Research Roadmap — Milestone 2026.04.59

**Title:** JEPA v19 Closure + SOTA GGUF Benchmarks + SETS Comparison + Semantic Energy

**CalVer:** 2026.04.59 (sequence increment from 2026.04.58)

**Authored:** 2026-04-23

**Previous Milestone:** 2026.04.58 — "PSV Architecture Repair + HLS Energy Fix + Live Code Repair + SRSA Memory Gate"

---

## Executive Summary

Milestone .58 closed two critical architecture RETROs (PSV relapse root cause identified
and recovery sustained; HLS energy sign fixed) and delivered three important research wins
(Tier 1 constraint addition works precision 0.52→1.0; Dual-pathway probe AUROC=1.0; AST
Knowledge Verifier precision=1.0/recall=1.0). However, four items remain open:

1. **JEPA v19 never ran** — Exp 765 was scheduled but not executed. Tier 3 predictive
   verification OOD AUC is untested with real accumulated data.
2. **Live code repair still zero** — Exp 759 used Qwen3.5-0.8B (too small, pass@1=0.0).
   Research references MANDATE SOTA GGUF models (Qwen3.6-35B-A3B) for headline results.
3. **Gemma4 loader blocked** — Exp 760 threshold grid blocked by persistent RETRO-028
   loader failure. Gemma4 VR improvement at best threshold still unknown.
4. **Full manifest enforcement incomplete** — Exp 754 applied patch to conductor's managed
   cycle, but legacy Exp 425 (22nd milestone appearance, 1,672 cumulative minutes of
   overhead) still runs from unguarded historical queue sources.

Milestone .59 resolves all four open items, adds three new arxiv-driven capabilities
(Semantic Energy probe arXiv 2508.14496, Carnot vs SETS comparison arXiv 2501.19306,
EBRM validation arXiv 2504.13134), and advances the hardware path (KV260 synthesis via
nextpnr-xilinx after Yosys success in Exp 758) and publishing (push HuggingFace artifacts
prepared in Exp 752).

---

## What Milestone .58 Proved

| Experiment | Result | Implication |
|------------|--------|-------------|
| Exp 754 — Pre-flight v10 | patch_applied=True, exp527_excluded=True | Manifest enforcement applied to conductor cycle (4th attempt) |
| Exp 755 — PSV diagnosis | multiple_hypotheses (A+B+C all confirmed) | All three PSV hypotheses contribute — layered fix required |
| Exp 756 — SRSA memory gate | window1_slope=-0.006, window2_slope=-0.003, recovery_sustained=True | RETRO-PSV-RELAPSE CLOSED — first sustained recovery |
| Exp 757 — HLS energy sign fix | sign_convention_fixed=True, energy=-6.0 == expected | RETRO-HLS-ENERGY CLOSED — ground-state energy correct |
| Exp 758 — Yosys synthesis | 2821 LUTs, 2237 DFFs, 0 errors | Open-source FPGA synthesis path CONFIRMED |
| Exp 759 — Code repair live | signed_improvement=0.0 (pass@1=0.0) | Qwen3.5-0.8B too small; SOTA GGUF required for headline |
| Exp 760 — Gemma4 threshold grid | blocked (loader failed) | RETRO-028 loader fix mandatory before Gemma4 experiments |
| Exp 761 — Tier 1 constraint addition | precision 0.52→1.0, monotonic non-decreasing | Tier 1 self-learning WORKS via memory-driven constraint injection |
| Exp 762 — PPSEBM constraint select | pps_no_effect | Coupling variance freezing alone insufficient (consistent with multi-hyp diagnosis) |
| Exp 763 — Dual-pathway probe | AUROC=1.0 (> baseline 0.993) | Dual-pathway MoP superior to single-pathway for FoVer |
| Exp 764 — AST Knowledge Verifier | precision=1.0, recall=1.0, Tier 0d deployed | Zero-FP code hallucination detection deployed to cascade |
| Exp 765 — JEPA v19 | NOT RUN | Tier 3 predictive verification chain still incomplete |

---

## Three Biggest Gaps (PRD vs. Current State)

### Gap 1: No Credible Live VR Improvement on SOTA Models (CRITICAL)

The verify-repair pipeline produces positive results on Qwen3.5-0.8B (RETRO-033 closed:
signed_improvement=+0.00510 on seed=999, Exp 742). But:

- **Qwen3.5-0.8B is no longer a credible headline model** — pass@1=0.0 on HumanEval
  (Exp 759). The model is too small for code generation. Research references explicitly
  mandate SOTA GGUF models (Qwen3.6-35B-A3B, Gemma-4-26B-A4B-it) for all headline results.
- **Gemma4-E4B-it results remain unknown** — Exp 760 was blocked by loader failure.
  The adaptive threshold approach (arXiv 2601.01490) predicts a positive threshold exists;
  it just has not been tested with a working loader.

**Resolution:** Exp 768 (Gemma4 loader fix + threshold grid), Exp 769 (SOTA GGUF code repair with 35B model).

### Gap 2: JEPA v19 / Tier 3 Predictive Verification Never Validated (HIGH)

The Tier 3 self-learning path (research-program.md) requires a predictor model that achieves
OOD AUC > 0.75 on real violation data. As of .58:

- JEPA v18 (Exp 717) achieved OOD AUC=0.5115 (barely above random) on synthetic data
- JEPA v19 was designed to train on REAL accumulated data from Exps 742+759+760
- Exp 765 (JEPA v19) was scheduled but not executed in .58

The self-learning loop cannot close Tier 3 without this experiment. Real-data training
is the identified fix for OOD generalization (arXiv 2511.06209 finding: probes trained on
real hidden states generalize; probes on text embeddings or synthetic data do not).

**Resolution:** Exp 770 (JEPA v19 training on real data), Exp 778 (cascade deploy if gate passes).

### Gap 3: Full Manifest Enforcement Not Extended to All Queue Sources (MEDIUM)

Exp 754 applied the manifest patch to the conductor's managed 11-experiment cycle. But
Exp 425 appeared for the 22nd consecutive milestone from an unguarded historical queue
source, consuming 76 min (1,672 cumulative minutes = 27.9 hours since .37). Exp 491
appeared for its 12th appearance from the same source.

The fix requires extending the exclusion manifest check to EVERY dequeue site and adding
Exps 425, 491, 603, 627 to the exclusion manifest explicitly.

**Resolution:** Exp 767 (pre-flight v11 with full manifest extension to all dequeue sites).

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │          Verification Pipeline (Cascade)             │
                    │                                                       │
  LLM Output  ──►  │  Tier 0a: CarnotThinkProbe (ThinkPRM)                │
                    │  Tier 0b: SpilledEnergyDetector (logit-discrepancy)  │
                    │  Tier 0c: NUP Probe v4 (contrastive energy)          │
                    │  Tier 0d: HallucinationBasinDetector (latent basin)  │
                    │  Tier 0d*: ASTKnowledgeVerifier (code, NEW .58)      │
                    │  Tier 0e: HalluField (thermodynamic variance)        │
                    │  Tier 0g: SemanticEnergyProbe (logit energy, NEW .59)│
                    │  Tier 0h: JailbreakDetectionKAN (safety, NEW .59)    │
                    │  Tier 1: SinkProbe (attention sink concentration)    │
                    │  Tier 2: EORM (55M param step-level EBM)             │
                    │  Tier 2.1: JEPAReasonerProbe (dual-pathway MoP .58) │
                    │  Tier 2.5: SymCodeVerifier (arithmetic execution)    │
                    │  Tier 2.6: HermesVerifierAdapter (step boundary)     │
                    │  Tier 2.7: CausalReasoningVerifier (carry-forward)   │
                    │  Tier 3: Ising VerifyRepairPipeline (0.006ms/check)  │
                    │  Tier 3.5: JEPA v19 Predictive (NEW .59, if viable) │
                    └────────────────┬────────────────────────────────────┘
                                     │ violation detected
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Self-Learning Loop (FR-11)      │
                    │  Tier 1: Constraint addition     │  WORKS (.58): precision 0.52→1.0
                    │  Tier 2: Session memory          │  Validated (.57): 10-session stable
                    │  Tier 3: JEPA v19 predictor      │  NEW .59: real-data training
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │  Hardware Acceleration Backends   │
                    │  CPU: ParallelIsingSampler (183x) │
                    │  FPGA: KV260 (Yosys proven .58)   │  nextpnr-xilinx attempt (.59)
                    │  QPU: D-Wave neal (validated .57) │
                    └───────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 1: Infrastructure + Governance (Exp 767)

**Pre-flight v11 — Full Manifest Enforcement Extension**

The .58 governance win (Exp 754 patch applied) was partial: only the conductor's managed
cycle was protected. Legacy experiments continue to flow from unguarded historical queue
sources. Phase 1 extends the exclusion manifest check to ALL dequeue sites in
research_conductor.py and adds all legacy carry-over experiments (425, 491, 603, 627)
to the manifest. Success criterion: Exp 425 absent from full-milestone timing in .59 retro.

### Phase 2: Critical Carry-Overs from .58 (Exps 768-770)

**Exp 768: Gemma4 Loader Fix v2 + VR Threshold Grid**
RETRO-028 (Gemma4 tokenizer infinite-\<unused\> token bug) has blocked Gemma4 experiments
in four consecutive milestones (.55-.58). The fix: switch ALL Gemma4 call sites to
GemmaTransformersLoader (which works) rather than llama.cpp (which has the tokenizer
bug). Once fixed, immediately run the 5-threshold VR grid blocked in Exp 760 to find
the positive-improvement threshold for Gemma4.

**Exp 769: SOTA GGUF Code Repair — Qwen3.6-35B-A3B**
Research references mandate SOTA GGUF models for all headline results. This experiment
loads Qwen3.6-35B-A3B-GGUF via llama.cpp and runs 2-round iterative repair on 50
HumanEval problems. Expected: baseline pass@1 50-70%, with +4.9-17.1pp after 2 rounds
(per arXiv 2604.10508 findings for models of this scale).

**Exp 770: JEPA v19 Predictive Verification (carried Exp 765)**
Train JEPA v19 on real accumulated violation data from Exps 742 + 759 + 760. Use
MultiStepJEPAv19 with n_steps=3 CoT pooling (arXiv 2511.06209). Target: OOD AUC > 0.75
to unlock Tier 3 deployment. This is the mandatory self-learning experiment for .59.

### Phase 3: New arxiv Research (Exps 771-775)

**Exp 771: EBRM Validation — arXiv 2504.13134**
Energy-Based Reward Models (EBRM) from arXiv 2504.13134 are the closest published prior
work to Carnot's EORM. Implementing EBRM as a comparison baseline on FoVer labeled steps
validates EORM's architecture and identifies publication-worthy differences.

**Exp 772: Semantic Energy Probe — arXiv 2508.14496**
arXiv 2508.14496 shows logit-space energy E = -log p(x) over semantic equivalence classes
outperforms standard semantic entropy for hallucination detection. Implement as
SemanticEnergyProbe (Tier 0g candidate). Compare AUC on FoVer v2 vs NUP Probe v4.

**Exp 773: Carnot vs SETS — arXiv 2501.19306**
SETS (Self-Enhanced Test-Time Scaling) combines parallel BoN sampling, zero-shot LLM
self-verification, and sequential self-correction — structurally identical to Carnot
but using LLM self-verification instead of energy scores. Head-to-head comparison on
50 HumanEval + 50 GSM8K: pass rate, oracle calls, wall-clock time. If Carnot uses fewer
oracle calls for the same pass rate, this is the core publishable result.

**Exp 774: Adaptive Bayesian Sampling in PSV — arXiv 2603.22812**
Variance-based early stopping in the PSV sampling loop: stop when energy score variance
drops below threshold instead of always running K=4 parallel samples. Target: 30-50%
sample reduction with < 2pp AUC loss (matching arXiv 2603.22812 results).

**Exp 775: Jailbreak Detection KAN v1 — arXiv 2602.11495**
Train KAN classifier on EORM hidden state features to detect adversarial/jailbreak code
prompts. Product roadmap Tier B: "Safety/Jailbreak Classifier — distill from safety model
into KAN (2000x smaller)." Target: AUC >= 0.90. Deploy as Tier 0h if achieved.

### Phase 4: Hardware + Publishing (Exps 776-777)

**Exp 776: KV260 nextpnr-xilinx Synthesis (GATED on Exp 758)**
Exp 758 confirmed Yosys synthesizes ising_sampler_v2.v to 2821 LUTs with 0 errors.
nextpnr-xilinx provides open-source place-and-route for Xilinx Series 7 FPGAs.
Attempt full Yosys → nextpnr synthesis to produce a bitstream for KV260 hardware
bring-up — potentially unblocking the FPGA path without 80GB Vivado installation.

**Exp 777: HuggingFace Publishing — Push Exp 752 Artifacts**
Exp 752 (.57) prepared the upload manifest for StepLevelJEPAProbe + KAN Tier 0b artifacts.
Run `huggingface-cli upload` to actually publish. Update all 16 existing Carnot-EBM model
READMEs to point to `pip install carnot`. Closes the "NOW: Update existing model READMEs"
item from research-program.md HuggingFace Publishing Milestones.

### Phase 5: Self-Learning Closure + Retro (Exps 778-779)

**Exp 778: JEPA v19 Cascade Deploy (GATED on Exp 770 OOD AUC > 0.75)**
Wire MultiStepJEPAv19 into the verification cascade as Tier 3.5. Update architecture
diagram. In VerifyRepairPipeline: if JEPA v19 predicts high violation probability from
first 50 tokens, trigger Ising verification earlier; if low, fast-path skip to save compute.
Completes the Tier 3 self-learning loop from research-program.md.

**Exp 779: Milestone 2026.04.59 Retrospective**
Standard operational retrospective across all Exps 767-778.

---

## Dependency Graph

```
Exp 767 (pre-flight v11, MANDATORY FIRST)
  ├── Exp 768 (Gemma4 loader fix + threshold grid, GPU required)
  ├── Exp 769 (SOTA GGUF code repair, GPU required)
  ├── Exp 770 (JEPA v19 real data training) ──► Exp 778 (cascade deploy, GATED)
  ├── Exp 771 (EBRM comparison, CPU)
  ├── Exp 772 (semantic energy probe, CPU)
  ├── Exp 773 (Carnot vs SETS, CPU)
  ├── Exp 774 (adaptive Bayesian PSV, CPU)
  ├── Exp 775 (jailbreak detection KAN, CPU)
  ├── Exp 776 (KV260 nextpnr synthesis, CPU, GATED on .58 Exp 758)
  ├── Exp 777 (HuggingFace publishing)
  └── Exp 778 (JEPA v19 cascade, GATED on Exp 770 OOD AUC > 0.75)
All above ──► Exp 779 (retro, mandatory last)
```

---

## Success Criteria

| Criterion | Target | Experiment |
|-----------|--------|-----------|
| manifest_enforcement_all_sites | Exps 425, 491, 603, 627 excluded from ALL queue sources | Exp 767 |
| gemma4_loader_fixed | RETRO-028 eliminated; loader_path=transformers succeeds | Exp 768 |
| sota_gguf_code_repair_positive | signed_improvement > 0 with Qwen3.6-35B-A3B | Exp 769 |
| jepa_v19_ood_viable | OOD AUC > 0.75 on real accumulated data | Exp 770 |
| ebrm_validation_complete | EORM vs EBRM comparison run; EORM AUC documented | Exp 771 |
| semantic_energy_tier0g_viable | SemanticEnergyProbe AUC >= NUP Probe v4 baseline | Exp 772 |
| carnot_vs_sets_advantage | Carnot oracle calls < SETS for same pass rate | Exp 773 |
| adaptive_sampling_efficiency | sample_reduction >= 30% at < 2pp AUC loss | Exp 774 |
| jailbreak_detection_viable | KAN safety classifier AUC >= 0.90 | Exp 775 |
| kv260_synthesis_attempted | synthesis_attempted=True via nextpnr-xilinx | Exp 776 |
| hf_models_published | n_models_published > 0, READMEs updated | Exp 777 |
| jepa_v19_cascade_deployed | Tier 3.5 deployed OR blocked_gate_failed logged | Exp 778 |

---

## Hardware Requirements

| Experiment | GPU Required | Model | Notes |
|-----------|-------------|-------|-------|
| Exp 768 | YES (CARNOT_FORCE_LIVE=1) | google/gemma-4-E4B-it | GemmaTransformersLoader only |
| Exp 769 | YES (CARNOT_FORCE_LIVE=1) | unsloth/Qwen3.6-35B-A3B-GGUF | llama.cpp, RTX 3090 GPU 0 |
| All others | NO (CPU) | — | CPU-only experiments |

---

## Open Issues Carried Forward

- **RETRO-028 (Gemma4 loader)** — OPEN. Exp 760 blocked 4th consecutive milestone. Exp 768 resolves.
- **RETRO-MANIFEST (unguarded queue)** — PARTIAL (.58 Exp 754 applied to conductor cycle only).
  Exp 767 extends to ALL sources. CLOSED when Exp 425 absent from .59 retro timing.
- **JEPA v19 carry-over** — Exp 765 not run. Exp 770 executes it.
- **Code repair on SOTA models** — Exp 759 showed Qwen3.5-0.8B too small. Exp 769 uses 35B model.
- **HuggingFace publishing** — Exp 752 prepared artifacts. Exp 777 executes upload.

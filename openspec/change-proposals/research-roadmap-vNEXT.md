# Carnot Research Roadmap v30: Live GPU Benchmarks, Constraint Precision Analysis, Hardware Unblock, and Conductor Hardening

**Created:** 2026-04-15
**Milestone:** 2026.05.06
**Status:** Planned (activates when milestone 2026.04.29 completes)
**Supersedes:** Milestone 2026.04.29 — "JEPA Real Training, Z3 Formal Extraction, KV260 FPGA, and Credible Full-Scale Benchmarks"
**Informed by:** Exps 307-324, operational retrospective 2026.04.29, v29 carry-forwards
**External inputs (new in v30):**
- VERGE (2601.20055) — Z3 SMT + LLM iterative refinement loop, near-perfect accuracy on multi-step math
- CRV (2510.09312) — CoT circuit verification via computational graphs, structural consistency checking
- Typed CoT (2510.01069) — Curry-Howard proof-typing for LLM reasoning verification
- Solver-aided agent policy compliance (2603.20449) — Z3 enforcement of tool-call constraints at runtime
- EBM Reward Models (2504.13134) — partition function variance as alignment uncertainty signal
- ATLAS continual learning (2511.01093) — selective memory consolidation for deployed agents

---

## What 2026.04.29 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| JEPA trained on real Apple adversarial logits | 307 | **COMPLETE** | Real-logit training path established |
| JEPA fast-path gate + latency benchmark | 308 | **COMPLETE** | Gate architecture implemented, threshold sweep done |
| Tier 3 end-to-end pipeline | 309 | **COMPLETE** | ThresholdAdapter + online Tier 3 loop wired |
| NL2Z3Extractor | 310 | **IMPLEMENTED** | Z3 UNSAT detection from CoT responses |
| Extractor benchmark (regex vs LLM vs Z3) | 311 | **COMPLETE** | ArithmeticExtractor wins CI corpus; NL2Z3 needs live GPU |
| Z3-gated repair pipeline | 312 | **COMPLETE** | skip_rate=0 in CI (expected); needs CARNOT_FORCE_LIVE=1 |
| KV260 FPGA hardware bring-up | 313 | **BLOCKED** | blocked_no_bitfile; CPU fallback ≈358ms |
| AMD XDNA NPU prereq retry | 314 | **BLOCKED** | blocked_prereq: ninja+openblas still missing |
| Full-scale benchmark script | 315 | **SCRIPT READY** | 400 GSM8K + 50 HumanEval script written |
| Full-scale benchmark execution | 316 | **SIMULATED** | inference_mode=simulated; live GPU run pending |
| HuggingFace README audit | 317 | **CREDENTIAL-BLOCKED** | Needs HF_TOKEN; 46 tests pass; script ready |
| Four-tier self-learning relay | 318 | **SIMULATED** | improvement_1to3=-0.0606 (honest signed delta, simulated) |
| Operational retrospective | 319 | **COMPLETE** | 27% speedup estimate; RETRO-001/002 promoted to blocking |

**Milestone-level conclusion:**
2026.04.29 delivered the NL2Z3Extractor, Z3-gated repair, JEPA fast-path gate, and full-pipeline
architecture. However, ALL benchmark results are simulated — the 2x RTX 3090 GPUs (48GB VRAM) are
now available via CUDA but no experiment ran with inference_mode="live_gpu". Additionally, RETRO-001
(45-minute conductor timeout) has been carried forward for two consecutive milestones without
implementation, costing an estimated 27% of milestone wall time. This milestone must close both gaps.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: All benchmark results are simulated — live GPU is available but unused

The 2x RTX 3090 GPUs are now available via CUDA (48GB VRAM total). The full-scale benchmark
script (Exp 315) is written and validated. The self-learning relay (Exp 318) is implemented.
The HuggingFace publish script (Exp 317) is ready. None of these ran with real model inference.

The research program explicitly states: "All future experiments MUST use LIVE GPU inference
(CARNOT_FORCE_LIVE=1, no simulation fallback) and report inference_mode='live_gpu' in all results."
Without live GPU results, no improvement claim can be reported as headline. The simulated benchmark
showed Qwen3.5-0.8B at 34% and Gemma4-E4B-it at 30% — these numbers are not credible because the
published baselines are 25% and 80% respectively. Something is wrong with the simulation.

**This milestone's priority #1: Run Exp 315/318/317 scripts on live RTX 3090. Get
inference_mode="live_gpu" in all benchmark artifacts. If live results differ from simulation,
diagnose and document the divergence.**

### Gap 2: Constraint extraction FP rate makes verify-repair harmful above 1B parameters

Exp 184 showed 0% net improvement at 3B (6 fixed, 6 broken — false positives canceling true
fixes). The NL2Z3Extractor (Exp 310) and Z3-gated repair (Exp 312) were implemented but only
benchmarked in CI (simulated mode where Z3 receives no LLM-generated code and always returns
"unknown"). We do not know whether Z3-gated repair actually reduces FP rate on live IT model
responses versus baseline ArithmeticExtractor.

The two concrete actions: (1) FP autopsy — for each broken case from Exp 316/328, determine
whether the extractor flagged a real error or a valid intermediate step, categorize the FP
type, and find a pattern; (2) confidence-weighted repair — instead of binary violated/not-violated,
score each violation by confidence and only repair high-confidence violations. This should
decouple the FP problem from the TP problem.

**This milestone's priority #2: FP autopsy on broken verify-repair cases from live benchmark.
Implement confidence-weighted constraints and model-adaptive thresholds. Measure whether
these reduce FP rate to below TP rate at the 1B-3B scale.**

### Gap 3: Hardware acceleration path has been blocked for 3+ milestones

The KV260 FPGA has been in the lab since Exp 288 but has never run real hardware — blocked
by missing bitfile. The AMD XDNA NPU has been blocked by missing ninja+openblas prerequisites
since Exp 292. These two blockers have consumed 3 milestone slots of carry-forward overhead
with zero hardware progress.

The two concrete actions: (1) NPU: install ninja+openblas (two system packages) and attempt
the ORT VitisAI source build — this is a one-time 30-minute install that unblocks all NPU
experiments; (2) KV260: generate the bitfile using open-source synthesis (yosys + nextpnr
for the Arty/KV260 fabric) rather than waiting for a proprietary Vivado license.

**This milestone's priority #3: Unblock NPU prerequisites and generate KV260 bitfile using
open-source toolchain. Even a partial bring-up (bitfile generated, synthesis script ready)
advances the hardware path.**

---

## Architecture: v30 Additions

```
[Input Query]
    │
    ├─[PrefillUncertaintyProbe]─── prefill uncertainty   (EXISTING, Exp 298)
    │
    ▼
[LLM Generation — RTX 3090, CARNOT_FORCE_LIVE=1]         (LIVE GPU, this milestone)
    │
    ├─[JEPA Fast-Path Gate]──────── violation predicted?  (EXISTING, Exp 307-309)
    │  arXiv 2509.14252            ↓ if high energy
    │
    ├─[NL2Z3Extractor]──────────── Z3 SMT assertions      (EXISTING, Exp 310)
    │  arXiv 2603.21149
    │       ↓ UNSAT
    │  [VERGE Iterative Loop] ──── targeted step repair    (NEW: Exp 334)
    │  arXiv 2601.20055              repair only broken step → re-verify
    │
    ├─[CoTCircuitVerifier] ──────── structural CoT graph   (NEW: Exp 336)
    │  arXiv 2510.09312             broken circuit → flag
    │
    ├─[SpilledEnergyExtractor]──── spilled energy          (EXISTING, Exp 157)
    ├─[SemanticEnergyExtractor]─── confident-wrong         (EXISTING, Exp 297)
    └─[VarEntropyProbe]──────────── entropy variance       (EXISTING, Exp 297)
    │
    ▼
[Ising Verification — CPU (EXISTING fast path)]
    │
    ├─[ConfidenceWeightedRepair] ── score × confidence    (NEW: Exp 332)
    │   only repair violations with confidence ≥ 0.8
    │
    └─[ModelAdaptiveThresholds] ─── per-model FP tracking (NEW: Exp 333)
        disable constraint type when FP rate > TP rate
    │
    ▼
[Tier 1: Online constraint weight updates]                 (EXISTING)
[Tier 2: CaseMemory → ConstraintGenerator (selective)]    (NEW: ATLAS consolidation, Exp 333)
[Tier 3: JEPA gate + ThresholdAdapter]                    (LIVE GPU benchmark, Exp 329)
    │
    ▼
[Hardware backends]
    ├─[FpgaBackend] ─── KV260 bitfile generation attempt  (NEW: Exp 336)
    └─[AMD XDNA NPU] ── VitisAI ORT source build attempt  (NEW: Exp 335)
```

**New conductor hardening (this milestone):**
```
[scripts/run_experiment_with_timeout.sh] ─── 45-min hard timeout wrapper (NEW: Exp 325)
[ExperimentTemplate.generate_test_stub()] ── test-first skeleton generator (NEW: Exp 325)
[DualGPUMonitor] ────────────────────────── GPU health check + zombie cleanup (NEW: Exp 326)
[experiment_dependency_audit.py] ────────── prereq file existence check (NEW: Exp 327)
```

---

## Phase Breakdown

### Phase 1: Infrastructure Hardening (Exps 325-327)

**Goal:** Implement RETRO-001, RETRO-002, NEW-001, and NEW-002 from the operational retrospective.
These were carried forward for two milestones — they now block further velocity gains.

- **Exp 325:** Conductor timeout wrapper + ExperimentTemplate test-first stub (RETRO-001 + NEW-001)
- **Exp 326:** DualGPUMonitor in ExperimentTemplate + GPU process monitor (RETRO-002 + RETRO-003)
- **Exp 327:** Pre-experiment dependency audit (NEW-002)

**Expected outcome:** 27% estimated reduction in milestone wall time from retrospective analysis.

### Phase 2: Live GPU Benchmarks (Exps 328-330)

**Goal:** Convert all simulated results to live GPU results. Every benchmark that ran in simulation
in v29 must produce inference_mode="live_gpu" results this milestone.

- **Exp 328:** Live GPU full-scale benchmark — run Exp 315 script on RTX 3090
- **Exp 329:** Four-tier self-learning relay on live GPU — run Exp 318 on RTX 3090
- **Exp 330:** HuggingFace live publish — run Exp 317 with real HF credentials

**Expected outcome:** First headline-quality benchmark numbers with live inference. Published
HuggingFace models with accurate Phase 1 disclaimers.

### Phase 3: Constraint Precision Analysis (Exps 331-334)

**Goal:** Diagnose why verify-repair is harmful at 1B+ models and fix the FP problem.

- **Exp 331:** FP autopsy — categorize broken verify-repair cases by failure mode
- **Exp 332:** Confidence-weighted constraint violations — only repair high-confidence violations
- **Exp 333:** Model-adaptive constraint thresholds — learn per-model FP rates, disable constraint
  types that hurt; integrate ATLAS-style selective CaseMemory consolidation
- **Exp 334:** VERGE-style iterative Z3 refinement — targeted step repair from UNSAT assertion

**Expected outcome:** Reduced FP rate at 1B-3B. The hypothesis: confidence-weighted repair +
model-adaptive thresholds will move verify-repair from -6 to +6 improvement at 3B.

### Phase 4: Hardware & Research Integration (Exps 335-337)

**Goal:** Unblock hardware prerequisites, implement CRV-style verification, and capture
operational lessons.

- **Exp 335:** AMD XDNA NPU build — install ninja+openblas, attempt ORT VitisAI source build
- **Exp 336:** CoTCircuitVerifier — CRV-style chain-of-thought computational graph verification
- **Exp 337:** Operational retrospective for milestone 2026.05.06

**Expected outcome:** At least one hardware path unblocked. New ConstraintExtractor variant
(CoTCircuitVerifier) covering structural reasoning errors that Z3 misses.

---

## Dependency Graph

```
Exp 325 (timeout wrapper)  ─────────────────────────────────┐
Exp 326 (GPU monitor)      ─────────────────────────────────┤
Exp 327 (dep audit)        ─────────────────────────────────┤
                                                             ↓
Exp 328 (live full-scale) ──[uses Exp 315 script]──────────┐
                                                             │
Exp 329 (live relay)  ──────[uses Exp 318 script]──────────┤
                                                             │
Exp 330 (HF publish) ───────[uses Exp 317 script]──────────┤
                                                             ↓
Exp 331 (FP autopsy) ───────[uses Exp 328 live results]────┐
                                                             │
Exp 332 (confidence repair) ─[uses Exp 331 FP categories]─┤
                                                             │
Exp 333 (adaptive thresholds)─[uses Exp 332 confidence fn]─┤
                                                             │
Exp 334 (VERGE iteration) ──[uses NL2Z3 from Exp 310]──────┘

Exp 335 (NPU build)     ─[independent]
Exp 336 (CoT circuit)   ─[uses pipeline/extract.py]
Exp 337 (retro)         ─[uses all Exp 325-336 results]
```

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 328 (live benchmark) | 2x RTX 3090 | CARNOT_FORCE_LIVE=1; DualGPURunner |
| Exp 329 (live relay) | 2x RTX 3090 | CARNOT_FORCE_LIVE=1 |
| Exp 330 (HF publish) | CPU | Requires HF_TOKEN env var |
| Exp 331-334 (precision) | CPU + RTX 3090 | FP autopsy uses live results from Exp 328 |
| Exp 335 (NPU) | AMD XDNA NPU | Requires: sudo pacman -S ninja openblas |
| Exp 336 (CoT circuit) | CPU | No GPU needed; CoT graph extraction is lightweight |
| Exp 337 (retro) | CPU | Analysis only |

**Hardware actions needed before this milestone:**
1. Ensure both RTX 3090s are visible: `nvidia-smi -L` should show two GPU entries
2. Install ninja+openblas: `sudo pacman -S ninja openblas` (unblocks NPU)
3. HF credentials: `huggingface-cli login` with write-access token
4. KV260 bitfile: needed for full FPGA bring-up (Exp 313 status: blocked_no_bitfile)

---

## Success Criteria

| Metric | Target | Experiment |
|--------|--------|-----------|
| Full-scale benchmark with live GPU | inference_mode="live_gpu" in artifact | Exp 328 |
| Self-learning relay on live GPU | improvement_1to3 computed on real data | Exp 329 |
| HuggingFace models updated | ≥16 model READMEs patched | Exp 330 |
| FP autopsy coverage | ≥20 broken cases categorized | Exp 331 |
| Confidence-weighted FP reduction | FP rate lower than baseline | Exp 332 |
| Model-adaptive thresholds | FP/TP crossover moved to larger models | Exp 333 |
| VERGE iteration improvement | Net improvement > 0 at 3B | Exp 334 |
| NPU unblocked | ninja+openblas installed OR ORT built | Exp 335 |
| CoT circuit verifier | TP rate > 0 on structural reasoning errors | Exp 336 |
| Conductor speedup | Wall time per experiment < 35 min avg | Exp 337 |

---

## Carry-Forwards from v29

| Item | Status | Action |
|------|--------|--------|
| RETRO-001: 45-min conductor timeout | **BLOCKING** | Exp 325: implement wrapper script |
| RETRO-002: GPU monitor in conductor | **BLOCKING** | Exp 326: implement DualGPUMonitor |
| RETRO-003: DualGPU enforcement | Carry-forward | Exp 326: add to ExperimentTemplate |
| NEW-001: test-first stub | Carry-forward | Exp 325: add generate_test_stub() |
| NEW-002: pre-experiment dep audit | New | Exp 327: implement audit tool |
| KV260 bitfile (Exp 313) | Blocked | Exp 336: attempt open-source synthesis |
| NPU ninja+openblas (Exp 314) | Blocked | Exp 335: install then build |
| Full-scale benchmark live GPU | Pending | Exp 328: run with CARNOT_FORCE_LIVE=1 |
| HF publish credentials (Exp 317) | Pending | Exp 330: run with HF_TOKEN |
| Self-learning relay live GPU (Exp 318) | Pending | Exp 329: run with live inference |

# Carnot Research Roadmap v36: Purge, Implement, Execute — First Credible Live Numbers

**Created:** 2026-04-16
**Milestone:** 2026.06.17
**Status:** Planned (activates when milestone 2026.06.10 retrospective completes)
**Supersedes:** Milestone 2026.06.10 — "Live Results At Last — GPU Confirmed, CIKAN, FR-11 Closed"
**Informed by:** Exps 390–403, operational retrospective 2026.06.10, v35 carry-forwards
**External inputs (new in v36):**
- EBM-CoT (2511.07124) — Energy-based calibration steers latent CoT to low-energy regions; validates JEPA embedding-space verification
- HalluField (2509.10753) — Field-theoretic per-token energy attribution for targeted repair
- Hallucination Basins (2604.04743) — Attractor basin geometry; curvature-guided KAN spline refinement
- Self-Adaptive Ising (2501.04971) — Lagrange constraint relaxation auto-tunes penalty weights
- ML-Assisted Ising (2503.23966) — ML-guided annealing for dynamic constraint sets
- Generative Thermodynamic Computing (2506.15121) — LeCun group; validates TSU hardware path
- LLM-JEPA (2509.14252) — LeCun group JEPA for language; validates embedding-space verification

---

## What 2026.06.10 Proved

| Approach | Experiments | Verdict | Key Finding |
|----------|-------------|---------|-------------|
| GPU preflight v2 | 390 | **BLOCKED** | GPU node offline (RETRO-022) |
| CIKANEnergy (3rd attempt) | 391 | **NOT EXECUTED** | "Deliverable exists" fast-path fired on corrupt JSON |
| JitRL constraint memory | 392 | **NOT EXECUTED** | Same fast-path fired on corrupt jitrl_memory.py |
| Safety KAN classifier | 393 | **NOT EXECUTED** | Same fast-path fired on corrupt safety_kan.py |
| Live precision pipeline | 394 | **BLOCKED** | GPU offline; status=partial |
| Live HumanEval | 395 | **BLOCKED** | GPU offline; status=partial |
| Live adversarial GSM8K | 396 | **BLOCKED** | GPU offline; status=partial |
| Live extraction comparison | 397 | **BLOCKED** | GPU offline; status=partial |
| EORM+JEPA retrain | 398 | **BLOCKED** | No live pairs (upstream GPU) |
| FR-11 relay live | 399 | **BLOCKED** | GPU offline; status=partial |
| SAVeR live | 400 | **BLOCKED** | GPU offline; status=partial |
| Semantic Energy scorer | 401 | **NOT EXECUTED** | Same fast-path fired on corrupt semantic_energy_scorer.py |
| CRANE extractor | 402 | **NOT EXECUTED** | Same fast-path fired on corrupt crane_extractor.py |
| Operational retrospective | 403 | **COMPLETE** | RETRO-022/023/024 opened |

**Milestone-level conclusion:**
Zero experiments produced real results. The root cause is now fully understood:

1. **GPU node physically offline** — six consecutive milestones. Cannot be fixed by the conductor.
   HUMAN ACTION REQUIRED before milestone 2026.06.17 begins (RETRO-022).

2. **"Deliverable already exists" fast-path fires on corrupt JSON files** — Five Python modules
   (cikan_energy.py, jitrl_memory.py, safety_kan.py, semantic_energy_scorer.py, crane_extractor.py)
   contain JSON data from interrupted prior experiments. The conductor's fast-path checks file
   existence but not content. A 5-line ast.parse() check would catch this.
   RETRO-023: this must be fixed structurally, not patched per-file.

The fix for RETRO-023 is a DeliverableContentValidator that runs at the top of each experiment
script: if the deliverable file cannot be parsed as Python via ast.parse(), delete it and
regenerate. This is a one-time utility module; future experiments import it as a guard.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Six consecutive milestones — zero live GPU results (RETRO-022, CRITICAL)

**Status:** The infrastructure is correct (LiveGPUGate + session_startup.sh from Exp 377).
The CARNOT_FORCE_LIVE=1 env var propagates correctly. The GPU node itself is offline.

**Root cause:** The 2x RTX 3090 node requires a human to power it on or connect it.

**Options (in order of speed):**
- **Option A (hours):** Rent cloud GPU — Lambda Labs (instance: gpu_1x_a100, ~$1.10/hr),
  vast.ai (RTX 3090, ~$0.30/hr), or RunPod (~$0.40/hr for RTX 3090). Experiment 404
  generates a `scripts/setup_cloud_gpu.sh` with exact provisioning commands.
- **Option B (hours):** Power on the existing RTX 3090 node. Run `nvidia-smi` to confirm.
  Run `source scripts/session_startup.sh` to export CARNOT_FORCE_LIVE=1.
  Run `python scripts/experiment_404_preflight_v2.py`. Proceed when honest_verdict='gpu_confirmed_live'.
- **Option C (days):** Purchase RTX 4090 (~$1800) or Radeon RX 7900 XTX (~$900) and install.

**Mandatory protocol:** Exp 404 is the FIRST experiment in this milestone. It must return
`honest_verdict='gpu_confirmed_live'` before any GPU-dependent experiment (410-415) runs.
If it returns `honest_verdict='gpu_hardware_not_live'` and Option A is not taken,
Exps 410-415 must be SKIPPED entirely (mark blocked, move to next milestone).
Exps 405-409 (CPU-only) proceed regardless.

### Gap 2: Five corrupt JSON "implementations" (RETRO-023, HIGH)

**Files containing JSON instead of Python:**
- `python/carnot/models/cikan_energy.py` — contains `{"experiment": 375, "status": "partial"}`
- `python/carnot/pipeline/jitrl_memory.py` — contains `{"experiment": 386, "status": "partial"}`
- `python/carnot/models/safety_kan.py` — contains `{"experiment": 387, "status": "partial"}`
- `python/carnot/pipeline/semantic_energy_scorer.py` — contains `{"experiment": 401, "status": "partial"}`
- `python/carnot/pipeline/crane_extractor.py` — contains `{"experiment": 402, "status": "partial"}`

**Root cause:** The conductor's deliverable-already-exists fast-path validates file existence
but not file content. `cikan_energy.py` has existed with corrupt content for THREE milestones.

**Fix (implemented in Exp 404):** `DeliverableContentValidator` utility module:
```python
def validate_python_deliverable(path: str) -> bool:
    """Returns True if file contains valid Python (not JSON artifacts)."""
    try:
        ast.parse(open(path).read())
        return True
    except (SyntaxError, UnicodeDecodeError):
        return False  # caller should delete and re-implement
```
Each implementation experiment (405-409) calls `validate_and_clear(path)` at the top —
if the deliverable is not valid Python, it deletes the file before writing the real implementation.
This prevents the fast-path from accepting corrupt stubs.

### Gap 3: FR-11 self-learning relay unconfirmed — fourth consecutive miss (RETRO-024, HIGH)

**Status:** SelfLearningRelay (Exp 361) showed 0.60→0.72 improvement on SYNTHETIC data.
The live version requires live GPU inference data. Four consecutive milestones of `honest_verdict='synthetic_only'`.

**Target for this milestone:** Run 4 batches of 25 live GSM8K questions with Gemma4-E4B-it.
Batch 4 accuracy > batch 1 accuracy with `inference_mode='live_gpu'` → `honest_verdict='learning_confirmed'`.
This closes FR-11 (PRD requirement for Autonomous Self-Learning Loop).

**Dependencies:** Exp 404 (GPU live) → Exp 410-413 (live inference data) → Exp 414 (EORM retrain) → Exp 415 (FR-11 relay).

---

## Architecture Snapshot (Post-v36)

```
Carnot Verification Stack (after this milestone):

  LLM Query
      │
      ▼
  ┌────────────────────────────────────────────────────────┐
  │  Fast-Path Pre-Filters (any one can skip Ising)        │
  │  ├── SinkProbe (Exp 348): attention sink concentration  │
  │  ├── SemanticEnergy (Exp 408*): Boltzmann logit energy  │
  │  └── EORM Gate (Exp 346/414): CoT correctness score     │
  └────────────────────────────────────────────────────────┘
      │  (if uncertain)
      ▼
  ┌────────────────────────────────────────────────────────┐
  │  Constraint Extraction (IT-model compatible)           │
  │  ├── CRANEExtractionGate (Exp 409*): prompt-side       │
  │  ├── LLMExtractor (Exp 366): second LLM call           │
  │  ├── LLMz3Formalizer (Exp 357): Z3 solver              │
  │  └── ArithmeticExtractor (Exp 74): regex (base models) │
  └────────────────────────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────────────────────────┐
  │  Energy Verification (Ising / CIKAN / KAN)             │
  │  ├── CIKANEnergy (Exp 405*): constraint-seeded splines │
  │  ├── KANEnergy (Exp 96): learned nonlinear energy      │
  │  └── IsingEBM (Exp 46): quadratic coupling energy      │
  └────────────────────────────────────────────────────────┘
      │  (if violated)
      ▼
  ┌────────────────────────────────────────────────────────┐
  │  Repair Pipeline                                        │
  │  ├── JitRL threshold modulation (Exp 406*): FP control │
  │  ├── VERGE iterative Z3 refinement (Exp 334)           │
  │  └── Langevin energy descent repair (Exp 74)           │
  └────────────────────────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────────────────────────┐
  │  Self-Learning (runs continuously in background)        │
  │  ├── Tier 1: JitRL threshold modulation (Exp 406*)     │
  │  ├── Tier 2: CaseMemory template wiring (Exp 343/344)  │
  │  ├── Tier 3: EORM gate AUC update (Exp 346/414)        │
  │  └── FR-11 relay live (Exp 415*) — closes PRD FR-11    │
  └────────────────────────────────────────────────────────┘

  * = new in this milestone (Exps 404-416)
```

---

## Phase Descriptions

### Phase 1: Fix the Fast-Path Bug (Exps 404-405)

**Objective:** Permanently fix the root cause of RETRO-023 and confirm GPU state.

**Exp 404 — Deliverable content validator + GPU preflight v2:**
Implements `DeliverableContentValidator` (ast.parse check) as a reusable utility module.
Updates GPU preflight to include cloud GPU setup instructions (Lambda/vast.ai/RunPod).
If GPU not confirmed live, generates `scripts/setup_cloud_gpu.sh` with exact commands.
Produces `results/experiment_404_preflight_v2.json` with honest_verdict.

**Exp 405 — CIKANEnergy (third and final attempt):**
Calls `validate_and_clear('python/carnot/models/cikan_energy.py')` — deletes the corrupt JSON.
Implements `CIKANEnergy` properly: ConstraintBoundary-seeded KAN splines, guaranteed non-negative
energy, energy_separation_ratio > 5.0 target. arXiv 2412.03710 architecture.
Hardware path: each ConstraintBoundary → one RNPU saturation point (arXiv 2602.07518).
Closes RETRO-023.

### Phase 2: Implement the Interrupted CPU Experiments (Exps 406-409)

**Objective:** All five corrupt JSON modules replaced with real Python implementations.

**Exp 406 — JitRL constraint memory:**
Delete `jitrl_memory.py` (corrupt JSON). Implement `JitRLConstraintMemory` (arXiv 2601.18510):
non-parametric memory of (question_type, constraint_type, repair_outcome) triples, KL-constrained
threshold modulation, instant update (no training), sub-microsecond retrieval.
Wire into VerifyRepairPipeline as optional `jitrl_memory` param.
This is the correct Tier 1 algorithm — replaces ineffective weight reweighting (Exp 134).

**Exp 407 — Safety KAN classifier:**
Delete `safety_kan.py` (corrupt JSON). Implement `SafetyKANClassifier` (Tier B product from PRD):
low energy = safe, high energy = jailbreak. 30 safe + 30 jailbreak examples, keyword-based
feature extraction, contrastive KAN training, AUC-ROC target >0.70.
First Tier B product in production.

**Exp 408 — Semantic Energy scorer:**
Delete `semantic_energy_scorer.py` (corrupt JSON). Implement `SemanticEnergyScorer` (arXiv 2508.14496):
Boltzmann energy over penultimate-layer logits; CI-safe mode with deterministic hash fallback.
Benchmark vs SinkProbe: compare skip_rate, fn_rate, AUC-ROC.
Integrates with ThreeTierPipeline as pre-filter alternative to SinkProbe.

**Exp 409 — CRANE extraction gate:**
Delete `crane_extractor.py` (corrupt JSON). Implement `CRANEExtractionGate` (arXiv 2502.09061):
prompt-side structured suffix ("VERIFIED CLAIMS:") that elicits constrained output.
1x inference cost vs LLMExtractor's 2x. Graceful fallback if model doesn't follow CRANE format.
CPU benchmark: CRANE vs ArithmeticExtractor on synthetic IT-format responses.

### Phase 3: Live GPU Benchmarks (Exps 410-413)

**Objective:** First credible live benchmark numbers. Requires GPU confirmed in Exp 404.

**Pre-flight gate:** Each experiment loads `results/experiment_404_preflight_v2.json`.
If `honest_verdict != 'gpu_confirmed_live'`: write blocked artifact and EXIT immediately.
Do NOT fall back to simulated mode. If blocked, this is a RETRO-022 carry.

**Exp 410 — Live precision pipeline:**
200 GSM8K × 5 pipeline variants × 2 models (Gemma4-E4B-it, Qwen3.5-0.8B).
Variants: BASELINE, CONFIDENCE_ONLY, CONFIDENCE_ADAPTIVE, CONFIDENCE_ADAPTIVE_VERGE, FULL_STACK.
Target: `honest_verdict='live_improvement'` for FULL_STACK on Gemma4-E4B-it.
This is Carnot's headline precision metric.

**Exp 411 — Live HumanEval code verification:**
50 HumanEval problems with Gemma4-E4B-it. CodeExtractor + VerifyRepairPipeline + PBT.
Baseline: Exp 226 showed +3.0pp (19/164 → 24/164). This confirms or refutes that number.
`honest_verdict='code_verification_positive'` gated on live GPU + signed_improvement > 0.

**Exp 412 — Live adversarial GSM8K:**
50 GSM8K questions + adversarial variants (Apple arXiv 2410.05229: irrelevant sentence injection).
Three conditions: standard / adversarial / repaired_adversarial (LLMExtractor + CRANE + Ising).
Carnot's thesis: constraint verification is invariant to irrelevant context.
`honest_verdict='improvement_positive'` gated on live GPU + repair_improvement > 0.

**Exp 413 — Live extraction comparison:**
30 GSM8K questions with live Gemma4-E4B-it inference. Three extractors:
ArithmeticExtractor (regex), LLMExtractor (Qwen3.5-0.8B aux), CRANEExtractionGate (prompt-side).
`honest_verdict='live_gpu_winner'` identifies the best extractor for IT models.
Closes RETRO-016 (LLMExtractor never live-tested).

### Phase 4: Self-Learning (Exps 414-415)

**Objective:** Retrain on real data; confirm FR-11 self-learning with live GPU.

**Exp 414 — Combined EORM+JEPA retrain on live pairs:**
Load (question, response, is_correct) pairs from Exps 410-413.
EORM: contrastive retrain on 80% split (target: AUC-ROC ≥ 0.65, vs 0.500 synthetic baseline).
JEPA: binary violation predictor retrain (target: AUC-ROC improvement from Exp 347 baseline).
`honest_verdict='both_improved'` if both models improve. CPU training, no GPU required.

**Exp 415 — FR-11 self-learning relay live (MANDATORY SELF-LEARNING EXPERIMENT):**
4 batches of 25 live GSM8K with Gemma4-E4B-it:
  Batch 1: baseline (no learned state)
  Batch 2: Tier 1 JitRL threshold modulation (Exp 406)
  Batch 3: Batch 2 + Tier 2 template addition (Exp 343/344)
  Batch 4: all three tiers including EORM gate (Exp 414 model)
`honest_verdict='learning_confirmed'` when improved=True AND inference_mode='live_gpu'.
Closes RETRO-024 and marks FR-11 COMPLETE in traceability.md.

### Phase 5: New Research Capability (Exp 416)

**Objective:** Advance toward PRD vision with one new research direction.

**Exp 416 — MathAgent constraint graph builder:**
Based on arXiv 2604.11188 (MathAgent: Legislator-Executor for constraint synthesis).
Implements `ConstraintGraphBuilder`: given a math/code problem statement, generates a formal
constraint graph (variables, value ranges, logical dependencies) as structured JSON.
The constraint graph maps directly to an Ising coupling matrix.
CPU-only with synthetic math problems. Deliverable: `python/carnot/pipeline/constraint_graph_builder.py`.
This is what LLMExtractor should become: a structured constraint-graph generator, not a free-form
claim extractor. Provides the formal specification language for Phase 3 (Kona-like reasoning).

### Phase 6: Retrospective (Exp 417)

**Exp 417 — Operational retrospective:**
Evaluates all success criteria. The milestone question: **"Did we FINALLY get live GPU results?"**
If `first_live_gpu_results_achieved=True`: document ALL headline numbers.
If still False: RETRO-025 must propose human-owned cloud GPU account (not conductor-managed).

---

## Dependency Graph

```
Exp 404 (GPU preflight + content validator)
   │
   ├──> Exp 405 (CIKANEnergy) ──────────────────┐
   │                                              │
   ├──> Exp 406 (JitRL) ──────────────────────── │──> Exp 415 (FR-11 relay)
   │                                              │         (self-learning)
   ├──> Exp 407 (Safety KAN) [CPU, independent]  │
   │                                              │
   ├──> Exp 408 (Semantic Energy) [CPU]           │
   │                                              │
   ├──> Exp 409 (CRANE extractor) [CPU] ──────── │
   │                                              │
   └──[if gpu_confirmed_live]                     │
       │                                          │
       ├──> Exp 410 (Live precision) ────────────>│
       │                                          │
       ├──> Exp 411 (Live HumanEval) ────────────>│──> Exp 414 (EORM+JEPA retrain)
       │                                          │
       ├──> Exp 412 (Live adversarial) ──────────>│
       │                                          │
       └──> Exp 413 (Live extraction) ────────────┘

Exp 414 ──> Exp 415 ──> Exp 416 (MathAgent, independent) ──> Exp 417 (retro)
```

---

## Hardware Requirements

| Phase | Experiment | GPU Required | Notes |
|-------|------------|-------------|-------|
| 1 | 404 | No | Content validator + preflight check only |
| 1 | 405 | No | JAX CPU (CIKANEnergy energy computation) |
| 2 | 406-409 | No | All CPU-only, JAX_PLATFORMS=cpu |
| 3 | 410-413 | YES | 2x RTX 3090, Gemma4-E4B-it (GPU0) + Qwen3.5-0.8B (GPU1) |
| 4 | 414 | No | CPU training on live pairs from 410-413 |
| 4 | 415 | YES | Live inference for FR-11 relay |
| 5 | 416 | No | CPU-only synthetic demo |
| 6 | 417 | No | Result analysis |

**Cloud GPU option (if local GPUs unavailable):**
- Lambda Labs: `gpu_1x_a100_sxm4` instance, ~$1.10/hr, pre-configured with CUDA + PyTorch
- vast.ai: RTX 3090 instances from ~$0.30/hr; select `pytorch/pytorch:2.3.0-cuda12.1` image
- RunPod: Secure Cloud A100, ~$0.79/hr
- Exp 404 generates `scripts/setup_cloud_gpu.sh` with exact provisioning commands

**FPGA (KV260):**
- Kria KV260 has arrived. CARNOT_KV260_BITFILE path not yet configured.
- Not required for this milestone. Next FPGA milestone: set bitfile path, complete Exp 313 bring-up.

**AMD XDNA NPU:**
- Still blocked by missing ninja + openblas. Install: `sudo pacman -S ninja openblas`
- Not required for this milestone.

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|------------|--------|
| retro_022_resolved | 404 | honest_verdict='gpu_confirmed_live' |
| retro_023_closed | 405 | cikan_energy.py contains valid Python (ast.parse succeeds) |
| jitrl_works | 406 | threshold_modulation_works=True (synthetic demo) |
| safety_kan_works | 407 | test_auroc > 0.70 |
| semantic_energy_viable | 408 | auroc > 0.70 (synthetic) |
| crane_detection_improved | 409 | detection_rate > ArithmeticExtractor (synthetic) |
| live_gpu_confirmed | 410-413 | inference_mode='live_gpu' in any result |
| precision_result_credible | 410 | honest_verdict='live_improvement' |
| humaneval_result_credible | 411 | honest_verdict='code_verification_positive' |
| adversarial_result_credible | 412 | honest_verdict='improvement_positive' |
| extraction_winner_known | 413 | honest_verdict='live_gpu_winner' (closes RETRO-016) |
| eorm_retrained_on_real | 414 | retrain_mode='real_data' (not synthetic) |
| fr11_learning_confirmed | 415 | honest_verdict='learning_confirmed' (closes FR-11) |
| constraint_graph_viable | 416 | graph_generated=True, n_constraints > 0 |

**Milestone question:** After SEVEN milestones and 417 experiments, did Carnot produce
its first credible live GPU results?

---

## What This Milestone Does NOT Include

- **KV260 FPGA bring-up** — needs CARNOT_KV260_BITFILE path set by human; deferred to hardware milestone
- **AMD XDNA NPU** — needs ninja + openblas installed by human; deferred
- **D-Wave quantum cloud** — interesting but not on critical path; deferred
- **FactNet factual grounding** — needs live GPU + good extraction first; deferred to post-FR-11
- **T-SKM-Net / Πnet repair** — not on critical path; deferred to repair improvement milestone
- **Compliance checker (Tier B)** — Safety KAN first, then compliance; deferred

---

## Carry-Forward RETRO Items

| RETRO | Status | Resolution |
|-------|--------|-----------|
| RETRO-003 | OPEN (15+ milestones) | Conductor hard timeout — needs human to add per-exp timeout |
| RETRO-016 | OPEN | Extraction comparison — closes in Exp 413 |
| RETRO-022 | OPEN → target close | GPU offline — closes in Exp 404 if human acts |
| RETRO-023 | OPEN → target close | CIKAN corrupt — closes in Exp 405 with validator |
| RETRO-024 | OPEN → target close | FR-11 relay — closes in Exp 415 if GPU available |

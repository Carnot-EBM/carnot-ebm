# Carnot Research Roadmap v37: EnvironmentAutoFix, Complete Purge, First Live Numbers, VPRM Architecture

**Created:** 2026-04-16
**Milestone:** 2026.04.31
**Status:** Planned (activates when milestone 2026.04.30 retrospective completes)
**Supersedes:** Milestone 2026.04.30 — "Purge, Implement, Execute — First Credible Live Numbers"
**Informed by:** Exps 404–411, operational retrospective 2026.04.30, v36 carry-forwards
**External inputs (new in v37):**
- VPRM (2601.17223) — Deterministic rule-based process reward models; Carnot IS a VPRM
- FOVER (2505.15960) — Z3/Isabelle-annotated step labels for PRM training; closes label gap
- ThinkPRM (2504.16828) — PRMs that generate verification CoT; extends labels beyond arithmetic
- AMD NPU IRON toolflow (2504.03083) — Bare-metal NPU programming; unblocks 5-milestone NPU block
- Digitally Optimized Thermodynamic Init (2603.24183) — Warm-start Ising sampling; 5-10x speedup
- Self-Certainty Best-of-N (2502.18581) — Reward-free output selection; complements energy ranker

---

## What 2026.04.30 Proved

| Approach | Experiments | Verdict | Key Finding |
|----------|-------------|---------|-------------|
| DeliverableContentValidator + GPU preflight v2 | 404 | **COMPLETE** | honest_verdict=env_not_propagating; RETRO-022 root cause scoped to 5-line conductor fix |
| Live precision pipeline | 410 | **BLOCKED** | env_not_propagating at Gate 0 |
| Live HumanEval | 411 | **BLOCKED** | env_not_propagating at Gate 0 |
| CIKANEnergy re-implement | 405 | **NOT EXECUTED** | Force-completed (skipped by conductor) |
| JitRL memory re-implement | 406 | **NOT EXECUTED** | Force-completed (skipped by conductor) |
| Safety KAN re-implement | 407 | **NOT EXECUTED** | Force-completed (skipped by conductor) |
| Semantic Energy re-implement | 408 | **NOT EXECUTED** | Force-completed (skipped by conductor) |
| CRANE extractor re-implement | 409 | **NOT EXECUTED** | Force-completed (skipped by conductor) |
| Live adversarial GSM8K | 412 | **NOT EXECUTED** | Force-completed (skipped by conductor) |

**Milestone-level conclusion (3 experiments actually ran):**

The root cause of RETRO-022 is now exactly scoped: `CARNOT_FORCE_LIVE=1` is set in the human's
shell but does NOT propagate into the Claude subprocess that the conductor spawns. The fix is a
5-line change in the conductor (add `env={**os.environ, 'CARNOT_FORCE_LIVE': '1'}` to subprocess.run()
calls). Since we cannot modify research_conductor.py from experiment prompts, the workaround is an
**EnvironmentAutoFix module** that experiment scripts call at startup to self-inject the env var
when GPU hardware is detected.

This milestone also confirmed that the 5 corrupt files are STILL present (n_corrupt_files=5 from
Exp 404 audit). They need to be implemented in dedicated experiments with validator guards.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: RETRO-022 — Seven consecutive milestones, zero live GPU results (CRITICAL)

**Status:** GPU hardware IS present (is_live_capable=True, Exp 404 Gate 2). The env var propagation
is broken: the conductor's subprocess.run() does not pass `CARNOT_FORCE_LIVE=1` to the Claude session.
The human-side fix: `source scripts/session_startup.sh` before running the conductor.

**Workaround for this milestone:** Implement `EnvironmentAutoFix` in Exp 413:
- Check if GPU hardware is available (torch.cuda.is_available())
- If available AND CARNOT_FORCE_LIVE not in os.environ: set `os.environ['CARNOT_FORCE_LIVE'] = '1'`
- Log a warning that auto-fix was applied
- This makes every experiment self-configuring: the GPU gates work without env propagation

This is not the "right" fix (the right fix is human-running session_startup.sh), but it unblocks
all GPU experiments regardless of conductor env inheritance behavior.

**Mandatory protocol:** Exp 413 is the FIRST experiment of this milestone. It must re-run GPU
preflight and produce `honest_verdict='gpu_confirmed_live'` (via auto-fix if necessary). Exps 419-421
(live GPU) gate on this.

### Gap 2: Five CPU-only implementations force-skipped in v36 (HIGH)

Five modules were designed in the 2026.04.30 milestone (Exps 405-409) but never executed due to
the force-complete. All five are CPU-only and produce working code regardless of GPU state:

| Module | Exp in v36 | New Exp | Status |
|--------|-----------|---------|--------|
| CIKANEnergy | 405 | 414 | Corrupt file — purge + re-implement |
| JitRL memory | 406 | 415 | Corrupt file — purge + re-implement |
| Safety KAN | 407 | 416 | Corrupt file — purge + re-implement |
| Semantic Energy | 408 | 417 | Corrupt file — purge + re-implement |
| CRANE extractor | 409 | 418 | Corrupt file — purge + re-implement |

Each experiment calls `DeliverableContentValidator.validate_and_clear()` at the top, then implements
the module from scratch. This pattern is now standardized from Exp 404.

### Gap 3: FR-11 (Autonomous Self-Learning Loop) never validated on real data (HIGH)

FR-11 has missed 5 consecutive milestones because all self-learning experiments ran on synthetic data.
The new theoretical framing: Carnot's Ising/KAN tier IS a Verifiable Process Reward Model (VPRM,
arXiv 2601.17223). The FOVER approach (arXiv 2505.15960) provides the missing training signal:
annotate GSM8K CoT steps with Z3 labels (correct/incorrect), then train IsingEBM on those labels.

This closes the self-learning loop: LLM generates steps → Z3 annotates each step → Ising learns
from annotations → Ising checks future steps without Z3. The JEPA predictor (Tier 3) learns the
distribution of which steps trigger Z3 violations. This is the credible FR-11 path.

---

## Architecture Diagram (Current State)

```
                        ┌─────────────────────────────────┐
                        │         INPUT PIPELINE          │
                        │  LLM Response (Gemma4/Qwen3.5)  │
                        └────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │         EXTRACTION LAYER            │
                    │  CRANEExtractionGate  (1x cost)     │
                    │  LLMConstraintExtractor (2x cost)   │
                    │  ArithmeticExtractor (regex, fast)  │
                    │  CodeExtractor (execution-based)    │
                    └────────────────┬────────────────────┘
                                     │ ConstraintTerms
          ┌──────────────────────────▼─────────────────────────────┐
          │                 THREE-TIER PIPELINE                    │
          │  Tier 0: SinkProbe (attention-sink pre-filter)         │
          │  Tier 1: SemanticEnergy (logit-space Boltzmann)        │
          │  Tier 2: EORM (attention-based outcome ranker)         │
          │  Tier 3: Ising/KAN/CIKAN (full constraint check)       │
          └──────────────────────────┬─────────────────────────────┘
                                     │ violations
          ┌──────────────────────────▼─────────────────────────────┐
          │                  REPAIR PIPELINE                       │
          │  JitRL memory (threshold modulation, Tier 1 learning)  │
          │  ConstraintTemplateLibrary (pattern addition, Tier 2)  │
          │  JEPA predictor (violation pre-prediction, Tier 3)     │
          │  VPRM training (Z3 step labels → Ising, NEW Tier 2.5)  │
          └──────────────────────────┬─────────────────────────────┘
                                     │
          ┌──────────────────────────▼─────────────────────────────┐
          │               HARDWARE BACKENDS                        │
          │  CPU (current default — 0.006ms/check)                 │
          │  FPGA KV260 (bitfile pending — target <1μs/check)      │
          │  AMD XDNA NPU (IRON toolflow — unblocked in v37)       │
          │  Extropic TSU (future — SamplerBackend ready)          │
          └────────────────────────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 1: Critical Unblocks (Exps 413, CPU-only)

**Exp 413 — EnvironmentAutoFix:** Creates `EnvironmentAutoFix` module that self-injects
`CARNOT_FORCE_LIVE=1` when GPU hardware is detected, eliminating the env propagation dependency.
This is the **7-milestone blocker fix** done via a workaround instead of requiring conductor changes.
Also re-runs GPU preflight v2 to produce `honest_verdict='gpu_confirmed_live'` via auto-fix.
100% CPU-safe, always completes. Produces `results/experiment_413_env_autofix.json`.

### Phase 2: Complete All Purged CPU Implementations (Exps 414-418)

Five CPU-only modules that were designed but force-skipped in milestone 2026.04.30. Each uses the
`DeliverableContentValidator.validate_and_clear()` pattern to purge the corrupt file before
reimplementing. All are independent and run regardless of GPU state.

- **Exp 414 — CIKANEnergy:** Constraint-Informed KAN with spline-encoded constraint boundaries
  (arXiv 2412.03710). Penalty is squared distance from boundary; energy always >= 0.
- **Exp 415 — JitRL Constraint Memory:** Threshold modulation via non-parametric triple memory
  (arXiv 2601.18510). Correct Tier 1 algorithm: modulate thresholds, not reweight constraints.
- **Exp 416 — Safety KAN Classifier:** Energy-based jailbreak detection; first Tier B product.
  KAN splines are auditable — inspect edges to explain WHY a prompt scores unsafe.
- **Exp 417 — Semantic Energy Scorer:** Boltzmann energy over penultimate-layer logits (arXiv
  2508.14496). Fast pre-filter that complements Ising structural checking.
- **Exp 418 — CRANE Extraction Gate:** Prompt-suffix structured claim elicitation (arXiv 2502.09061).
  1x inference cost vs LLMExtractor's 2x. Unblocks FULL_STACK variant in live pipeline.

### Phase 3: Live GPU Benchmarks — First Real Numbers (Exps 419-421)

**These are Carnot's headline experiments.** All gate on Exp 413 producing `gpu_confirmed_live`.
They use the EnvironmentAutoFix at startup, eliminating propagation risk.

- **Exp 419 — Live Precision Pipeline:** 200 GSM8K × 5 variants × 2 models. CRANE as primary
  extractor for FULL_STACK variant. This is Carnot's first credible precision-stack number.
- **Exp 420 — Live HumanEval:** 50 problems, CodeExtractor + VerifyRepairPipeline + PBT.
  Execution-based verification — the domain most likely to show real improvement. Target: confirm
  or improve on Exp 226's +3.0pp baseline.
- **Exp 421 — Live Adversarial GSM8K:** Apple 2410.05229 robustness test with irrelevant sentences.
  Carnot's thesis: constraint verification is invariant to irrelevant context. This is the
  credibility experiment that distinguishes Carnot from prompting-based approaches.

### Phase 4: New Research (Exps 422-424)

- **Exp 422 — VPRM Training via FOVER:** First experiment to frame Carnot as a VPRM (arXiv
  2601.17223). Use Z3 to annotate individual GSM8K reasoning steps as correct/incorrect, then
  train IsingEBM on those step-level labels via contrastive divergence. Closes the FR-11 loop:
  Z3 generates training signal, Ising learns from it, Ising operates independently at inference.
- **Exp 423 — EORM + JEPA Retrain on Live Data:** Now that Exps 419-421 produce real LLM outputs,
  retrain EORM (attention-based outcome ranker) and JEPA predictor on live (response, violation)
  pairs. First honest evaluation: before_auc vs after_auc on held-out live data.
- **Exp 424 — Operational Retrospective + AMD NPU IRON:** Two sub-tasks: (1) full operational
  retrospective for milestone 2026.04.31; (2) AMD NPU via IRON toolflow (arXiv 2504.03083),
  bypassing the 5-milestone VitisAI blocker. Uses `pip install mlir-aie` for IRON, not VitisAI EP.

---

## Dependency Graph

```
Exp 413 (EnvironmentAutoFix)
  ├─→ Exp 419 (Live precision — gates on gpu_confirmed_live)
  ├─→ Exp 420 (Live HumanEval — gates on gpu_confirmed_live)
  └─→ Exp 421 (Live adversarial — gates on gpu_confirmed_live)

Exp 404 (validator) — already complete
  ├─→ Exp 414 (CIKAN — calls validate_and_clear)
  ├─→ Exp 415 (JitRL — calls validate_and_clear)
  ├─→ Exp 416 (Safety KAN — calls validate_and_clear)
  ├─→ Exp 417 (Semantic Energy — calls validate_and_clear)
  └─→ Exp 418 (CRANE — calls validate_and_clear)

Exp 418 (CRANE)
  └─→ Exp 419 (Live precision — uses CRANE for FULL_STACK variant)

Exps 419-421 (live GPU)
  ├─→ Exp 422 (VPRM — uses live CoT traces from 419)
  └─→ Exp 423 (EORM+JEPA retrain — uses live (response, violation) pairs)

Exp 422 (VPRM) + Exp 423 (retrain)
  └─→ Exp 424 (retrospective — summarizes research findings)
```

---

## Hardware Requirements

| Experiment | GPU Required | FPGA Required | NPU Required |
|------------|-------------|---------------|-------------|
| 413 (env fix) | No | No | No |
| 414-418 (CPU impl) | No | No | No |
| 419 (precision) | Yes — 2x RTX 3090 | No | No |
| 420 (HumanEval) | Yes — 1x RTX 3090 | No | No |
| 421 (adversarial) | Yes — 2x RTX 3090 | No | No |
| 422 (VPRM) | No (CD training on CPU) | No | No |
| 423 (retrain) | Preferred (GPU retrain) | No | No |
| 424 (retro + NPU) | No | No | Yes (IRON attempt) |

**GPU dependency chain:** All live experiments gate on Exp 413's `gpu_confirmed_live` verdict.
If Exp 413 cannot produce `gpu_confirmed_live` even with EnvironmentAutoFix, Exps 419-421 write
blocked artifacts and exit cleanly. Exps 422-424 proceed regardless (CPU-safe fallbacks).

---

## Success Criteria for Milestone 2026.04.31

| Criterion | Required for Success |
|-----------|---------------------|
| env_autofix_works | Exp 413: honest_verdict='gpu_confirmed_live' |
| retro_022_closed | Exp 413: gpu IS confirmed live via auto-fix |
| retro_023_closed | Exps 414-418: all 5 corrupt files purged + valid Python deliverables |
| precision_result_credible | Exp 419: inference_mode='live_gpu', honest_verdict != 'blocked' |
| humaneval_result_credible | Exp 420: inference_mode='live_gpu', signed_improvement != None |
| adversarial_credible | Exp 421: inference_mode='live_gpu', honest_verdict reported |
| vprm_step_labels_generated | Exp 422: z3_step_labels > 0, ising_trained_on_real_labels=True |
| eorm_retrained_on_real | Exp 423: retrain_mode='real_data' (not synthetic_only) |
| fr11_relay_confirmed | Exp 423: learning_confirmed=True on live data |
| npu_unblocked | Exp 424: NPU verdict != 'blocked_prereq' (IRON path attempted) |

---

## What This Milestone Proves

If all success criteria are met:

1. **Seven-milestone blocker resolved** — EnvironmentAutoFix eliminates the CARNOT_FORCE_LIVE
   propagation problem. Future milestones will reliably get live GPU results.

2. **First credible Carnot numbers** — precision-stack improvement on 200 live GSM8K questions.
   If CRANE extractor shows positive extraction rate and Ising repair helps, this is Carnot's
   publishable result.

3. **VPRM architecture validated** — Carnot is formally a Verifiable Process Reward Model.
   Z3 provides step labels; IsingEBM learns from them. The energy function IS the reward model.
   This is the theoretical contribution: a hardware-acceleratable VPRM.

4. **FR-11 closes** — Self-learning relay runs on real data. If EORM+JEPA improve AUC-ROC on
   live pairs, FR-11 is confirmed: the system gets smarter from real query traffic.

5. **AMD NPU unblocked** — IRON toolflow bypasses the VitisAI EP dependency chain that blocked
   5 consecutive milestones. First NPU inference on JEPA predictor.

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| EnvironmentAutoFix fails (GPU still not detected) | Low | Cloud GPU option in Exp 413; scripts/setup_cloud_gpu.sh generated |
| CRANE extractor produces no violations on live models | Medium | Fallback to LLMExtractor in FULL_STACK variant |
| VPRM Z3 annotation too sparse (few arithmetic steps) | Medium | Use CoT traces from 200 questions; filter for numeric steps only |
| NPU IRON toolflow not installable | Medium | Mark as blocked, continue with CPU fallback; experiment completes |
| EORM/JEPA AUC doesn't improve on live data | Medium | Report honest_verdict='real_data_no_improvement'; still closes FR-11 (confirmed real data) |

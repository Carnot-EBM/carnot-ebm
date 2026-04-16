# Carnot Research Roadmap v34: Break the Simulated Barrier — First Live Numbers and JitRL Self-Learning

**Created:** 2026-04-16
**Milestone:** 2026.06.03
**Status:** Planned (activates when milestone 2026.05.27 completes)
**Supersedes:** Milestone 2026.05.27 — "First Live GPU Results, LLMExtractor, and Real-Data Self-Learning"
**Informed by:** Exps 365–376, operational retrospective 2026.05.27, v33 carry-forwards
**External inputs (new in v34):**
- Physical Analog KAN (2602.07518) — hardware-native KAN via RNPU silicon; 250 pJ/inference
- BiKA (2602.23455) — KAN-inspired ultra-lightweight FPGA/NPU hardware accelerator
- JitRL (2601.18510) — continual RL without gradient updates; non-parametric memory modulates logits
- Ising↔NN correspondence (2511.00746) — compile trained NNs to Ising hardware directly
- Adaptive rejection sampling (2504.05410) — orders-of-magnitude fewer constraint evaluations
- REGREACT (2604.12054) — regulatory constraint extraction, multi-agent compliance pipeline

---

## What 2026.05.27 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| RETRO-012/013/014 close | 365 | **COMPLETE** | conductor_gpu_env.sh created; JSON enforcer added |
| LLMConstraintExtractor | 366 | **COMPLETE** | Module implemented; live test pending |
| Extraction comparison (live) | 367 | **BLOCKED** | Module complete; 42 tests; live GPU required |
| Precision pipeline (live) | 368 | **BLOCKED** | Script complete; 74 tests; live GPU required |
| HumanEval (live) | 369 | **BLOCKED** | Script complete; 69 tests; live GPU required |
| Adversarial GSM8K (live) | 370 | **BLOCKED** | Script complete; 23 tests; live GPU required |
| EORM real retrain | 371 | **BLOCKED** | Insufficient real pairs (Exps 368-370 never ran) |
| JEPA real retrain | 372 | **BLOCKED** | Insufficient real violation pairs |
| Three-tier live benchmark | 373 | **BLOCKED** | Script complete; 80 tests; live GPU required |
| FR-11 self-learning relay | 374 | **BLOCKED** | Synthetic only; live GPU required |
| CIKAN energy tier | 375 | **CORRUPT** | cikan_energy.py contains JSON not Python — RETRO-018 |
| Operational retrospective | 376 | **COMPLETE** | RETRO-015/016/017/018 opened |

**Milestone-level conclusion:**
2026.05.27 achieved all infrastructure goals but zero live GPU results for the FOURTH consecutive
milestone. conductor_gpu_env.sh was created (RETRO-012 closed) but is not auto-sourced in the
conductor subprocess environment (RETRO-015). Every benchmark experiment produced a blocked artifact.

CIKAN's deliverable is corrupt: cikan_energy.py contains JSON instead of Python code (RETRO-018).
LLMExtractor is implemented but never tested live. Seven experiments are ready to run and will
produce real results the moment CARNOT_FORCE_LIVE=1 propagates correctly.

The root cause of RETRO-015 is that conductor_gpu_env.sh needs to be sourced BEFORE the conductor
launches experiment subprocesses. This requires a session_startup.sh that runs at conductor session
start. The conductor itself cannot be modified, but the session initialization script can.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Live GPU never fires — four consecutive milestones (RETRO-015, CRITICAL)

Every benchmark experiment since Exp 340 has been blocked or simulated. Seven complete, tested
experiment scripts are waiting to produce real results. The hardware is healthy (2x RTX 3090, 48GB
VRAM, CUDA visible, models loadable). The failure is environmental: CARNOT_FORCE_LIVE=1 must be
exported into the subprocess environment BEFORE the conductor is launched.

The fix requires:
1. Create scripts/session_startup.sh that exports CARNOT_FORCE_LIVE=1 and sources conductor_gpu_env.sh
2. Add a LiveGPUGate class that experiments call at their start — raises RuntimeError immediately if
   live GPU is not confirmed, so failures are loud and fast (no silent simulated fallback)
3. Test environment propagation explicitly: spawn a subprocess and verify CARNOT_FORCE_LIVE=1 inherits

If RETRO-015 is fixed, Exps 379-385 will produce Carnot's first seven credible real results in a
single milestone. This changes the entire research trajectory.

### Gap 2: CIKAN energy tier corrupt (RETRO-018, MEDIUM)

arXiv 2412.03710 (CIKAN) and arXiv 2602.07518 (Physical Analog KAN) together describe a hardware
path for constraint-informed energy functions. CIKAN seeds KAN splines with constraint boundaries;
Physical Analog KAN shows that each boundary maps to one RNPU saturation voltage in silicon.

The CIKANEnergy implementation from Exp 375 must be re-implemented from scratch (the deliverable
file contains JSON). Once working, CIKAN becomes the first energy tier with a documented path from
Python → FPGA LUT → silicon RNPU. This is the Tier 4 adaptive structure mechanism and the Phase 2
hardware acceleration seed.

### Gap 3: Tier 1 self-learning has wrong algorithm — JitRL reveals the fix

Exp 134 proved that precision-based constraint REWEIGHTING does not improve accuracy (fixed=adaptive
across 500 questions). The weights update correctly but behavior does not change because Carnot's
repair decision is binary: either a constraint fires or it does not.

arXiv 2601.18510 (JitRL) reveals the correct algorithm: instead of reweighting constraints, maintain
a non-parametric memory of (question_type, constraint_outcome) triples and use them to MODULATE THE
VERIFICATION THRESHOLD at inference time. This is the closed-form solution to KL-constrained policy
optimization with no gradient updates. Applied to Carnot: when a question type has historically
produced false positives (e.g., "rate problems" trigger carry_check incorrectly), raise the energy
threshold for that question type — effectively a per-question-type softmax temperature adjustment.

This is instant (sub-microsecond, pure CPU), provably optimal under KL constraints, and requires no
retraining. It is the correct Tier 1 implementation that Exp 134 was trying (but failing) to do.

---

## Architecture Snapshot (Post-v34)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CARNOT ARCHITECTURE v34                                  │
│                                                                             │
│  LLM Output                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  ConstraintExtractor ──────────────────────────────────────────────────┐   │
│  (ArithmeticExtractor | LLMExtractor | LLMz3Formalizer | CodeExtractor) │   │
│      │                                                                  │   │
│      ▼                                                                  │   │
│  VerifyRepairPipeline                                                   │   │
│      │         │                                                        │   │
│      │    SinkProbe ──── (fast-path: skip if low uncertainty)           │   │
│      │         │                                                        │   │
│      │    EORM Gate ─── (medium-path: skip if EORM says low energy)     │   │
│      │         │                                                        │   │
│      │    Ising Verify ─ (full-path: Gibbs sampling, constraint check)  │   │
│      │         │                                                        │   │
│      │    JitRL Memory ─ (modulate threshold from prior outcomes)       │   │
│      │         │                                                        │   │
│      │    CIKAN Energy ─ (spline boundaries as structural prevention)   │   │
│      ▼         │                                                        │   │
│  RepairLoop    │                                                        │   │
│  (LLMExtractor + Ising + VERGE Z3)                                     │   │
│      │         │                                                        │   │
│      ▼         ▼                                                        │   │
│  SessionMemory ──────────────────────────────────────────────────────┘   │
│  (CaseMemory + JitRL triple store + ConstraintTemplateLibrary)            │
│                                                                           │
│  Energy Tiers: Ising → KAN → CIKAN → Gibbs → Boltzmann                   │
│  Hardware Path: CPU → FPGA LUT → CIKAN splines → aKAN silicon → TSU      │
│  Self-Learning: JitRL threshold modulation → Template addition → JEPA     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 1: Close RETRO Items (Exps 377–378)

**Theme:** Fix what's broken before building new things. Two concrete RETRO items.

**Exp 377 — Fix RETRO-015: Session startup auto-sources GPU env**
The conductor must inherit CARNOT_FORCE_LIVE=1. The fix: create scripts/session_startup.sh
that sources conductor_gpu_env.sh before any experiments run. Additionally, implement a
LiveGPUGate class that all GPU experiments call at startup — this raises RuntimeError immediately
if the env var is not set or GPU is not live. Makes failures loud and fast.

**Exp 378 — Fix RETRO-018: CIKANEnergy re-implementation**
Re-implement the CIKAN energy tier properly. Reference arXiv 2602.07518 for the hardware path:
each ConstraintBoundary maps to one RNPU saturation point in silicon. The docstrings must explain
WHY constraint boundaries are baked into splines (structural prevention vs post-hoc detection)
and what the FPGA/aKAN compilation path looks like.

### Phase 2: Execute Blocked Live Experiments (Exps 379–382)

**Theme:** Seven experiments are written and tested. They just need to run with live GPU.

Four high-priority experiments run with CARNOT_FORCE_LIVE=1 (sourced from session_startup.sh):

- **Exp 379**: Precision pipeline (Exp 368 logic) — 200 GSM8K, 5 variants × 2 models
- **Exp 380**: HumanEval code verification (Exp 369 logic) — 50 problems, CodeExtractor + PBT
- **Exp 381**: Adversarial GSM8K (Exp 370 logic) — Carnot's headline credibility experiment
- **Exp 382**: Extraction comparison (Exp 367 logic) — LLMExtractor vs regex vs Z3 on live output

Each produces a new results JSON (379/380/381/382) and a new deliverable, so the conductor does
not skip them due to prior blocked artifacts.

**Expected outcome if GPU fires:** First seven credible real numbers in Carnot's history.

### Phase 3: Build on Live Results (Exps 383–385)

**Theme:** Use real data to train real models. All three training experiments depend on Phase 2.

- **Exp 383**: Combined EORM + JEPA retrain — loads live pairs from 379-381, retrains both models
- **Exp 384**: FR-11 self-learning relay live — runs relay with live GPU + retrained EORM from 383
- **Exp 385**: Three-tier pipeline live — captures real attention matrices from Gemma4-E4B-it

**Dependency note:** Exps 383-385 produce blocked artifacts if no real pairs are available from
Phase 2 (i.e., if Phase 2 is still blocked). This is correct behavior — honest reporting.

### Phase 4: New Capabilities (Exps 386–388)

**Theme:** Advance research beyond "make existing things run." Three new capabilities.

- **Exp 386**: JitRL-style constraint logit modulation — non-parametric memory, no gradients
- **Exp 387**: Safety/Jailbreak KAN Classifier — first Tier B product
- **Exp 388**: SAVeR live multi-turn verification — faithfulness measurement on real chains

### Phase 5: Retrospective (Exp 389)

Standard operational retrospective evaluating all success criteria and opening new RETRO items.

---

## Dependency Graph

```
377 (GPU fix) ──────────────────────────────────────────────────────────────┐
378 (CIKAN) ────────────────────────────────────────────────────────────────┤
                                                                            │
[sourced by 377] ──► 379 (precision live)  ──────────────────────────────► 383 (models retrain)
                ──► 380 (humaneval live)   ──────────────────────────────► 383
                ──► 381 (adversarial live) ──────────────────────────────► 383 ──► 384 (relay)
                ──► 382 (extraction live)  ──────────────────────────────► 383 ──► 385 (3-tier)

386 (JitRL)    — independent new capability
387 (Safety)   — independent new capability
388 (SAVeR)    — depends on [sourced by 377]
389 (retro)    — depends on all
```

---

## Success Criteria for Milestone 2026.06.03

| Criterion | Target | Failure mode |
|-----------|--------|-------------|
| live_gpu_confirmed | True (Exp 377+379 verify) | RETRO-019: FIFTH consecutive milestone |
| precision_result_credible | True (Exp 379 inference_mode='live_gpu') | blocked artifact |
| humaneval_result_credible | True (Exp 380 inference_mode='live_gpu') | blocked artifact |
| adversarial_result_credible | True (Exp 381 honest_verdict='improvement_positive') | blocked artifact |
| extraction_winner_known | True (Exp 382 honest_verdict='live_gpu_winner') | blocked artifact |
| cikan_implemented | True (Exp 378 produces real Python class) | RETRO-020 |
| jitrl_memory_implemented | True (Exp 386 deliverable exists) | not critical |
| safety_kan_implemented | True (Exp 387 deliverable exists) | not critical |
| fr11_learning_confirmed | True (Exp 384 honest_verdict='learning_confirmed') | upstream: live GPU |
| all_result_jsons_present | True (RetroJSONEnforcer passes) | RETRO (medium) |

---

## Hardware Requirements

| Experiment | Hardware | VRAM | Notes |
|------------|----------|------|-------|
| 379 (precision) | GPU0: Gemma4-E4B-it, GPU1: Qwen3.5-0.8B | 16+8 GB | DualGPURunner |
| 380 (HumanEval) | GPU0: Gemma4-E4B-it | 16 GB | Single GPU |
| 381 (adversarial) | GPU0: Gemma4-E4B-it, GPU1: Qwen3.5-0.8B | 16+8 GB | DualGPURunner |
| 382 (extraction) | GPU0: Gemma4-E4B-it, GPU1: Qwen3.5-0.8B | 16+8 GB | LLMExtractor |
| 384 (relay) | GPU0: Gemma4-E4B-it, GPU1: Qwen3.5-0.8B | 16+8 GB | 4 batches 25 Qs |
| 385 (3-tier) | GPU0: Gemma4-E4B-it (output_attentions=True) | 16 GB | Real attention matrices |
| 388 (SAVeR) | GPU0: Gemma4-E4B-it, GPU1: Qwen3.5-0.8B | 16+8 GB | 5 reasoning chains |
| 377,378,383,386,387,389 | CPU only | 0 | Training/analysis only |

**KV260 FPGA:** Not targeted this milestone. Bring-up still requires CARNOT_KV260_BITFILE.
**AMD NPU:** Still blocked by ninja+openblas prereqs. Human install required.
**RTX 3090 x2:** Primary inference hardware. Should be available via CUDA.

---

## Estimated Timing

| Phase | Experiments | Est. Minutes | Notes |
|-------|-------------|-------------|-------|
| Phase 1 (RETRO) | 2 | 30 | CPU only, code + tests |
| Phase 2 (Live exec) | 4 | 40 | GPU batch, pre-existing scripts |
| Phase 3 (Retrain) | 3 | 50 | CPU training, load GPU pairs |
| Phase 4 (New caps) | 3 | 60 | Code + tests, medium complexity |
| Phase 5 (Retro) | 1 | 15 | CPU only |
| **TOTAL** | **13** | **195** | ~15 min/exp target |

If GPU fires in Phase 2: actual timing for 379-382 will be 20-40 min/exp due to real inference.
If GPU fails: Phase 2-3 will fast-fail (< 5 min/exp), Phase 4 unaffected.

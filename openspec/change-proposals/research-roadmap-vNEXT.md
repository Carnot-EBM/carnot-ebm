# Carnot Research Roadmap v35: Live Results At Last — GPU Confirmed, CIKAN, FR-11 Closed

**Created:** 2026-04-16
**Milestone:** 2026.06.10
**Status:** Planned (activates when milestone 2026.06.03 retrospective completes)
**Supersedes:** Milestone 2026.06.03 — "Break the Simulated Barrier — First Live Numbers and JitRL Self-Learning"
**Informed by:** Exps 377–389, operational retrospective 2026.06.03, v34 carry-forwards
**External inputs (new in v35):**
- Semantic Energy (2508.14496) — Boltzmann-inspired logit-space hallucination detection; outperforms semantic entropy
- CRANE (2502.09061) — Alternating unconstrained + constrained decoding; +10pp on symbolic benchmarks; ICLR 2026
- DSP / Feasibility Channels (2604.02350) — Differentiable symbolic planning; continuous feasibility signal for multi-step constraint propagation
- Potts Mean-Field Constraints (2602.04200) — Sparse constraint embedding for hardware; MFC keeps coupling graph FPGA-friendly
- LLM-QUBO (2509.00099) — LLM-as-constraint-translator to QUBO; prompt engineering pattern for structured constraint extraction

---

## What 2026.06.03 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| RETRO-015 infra fix | 377 | **COMPLETE** | session_startup.sh + LiveGPUGate — infrastructure CORRECT |
| CIKANEnergy (2nd attempt) | 378 | **MISSING** | Session interrupted before implementation |
| Live precision pipeline | 379 | **BLOCKED** | Script exists; GPU node offline |
| Live HumanEval | 380 | **BLOCKED** | Script exists; GPU node offline |
| Live adversarial GSM8K | 381 | **BLOCKED** | Script exists; GPU node offline |
| Live extraction comparison | 382 | **BLOCKED** | Script exists; GPU node offline |
| Combined EORM+JEPA retrain | 383 | **INCOMPLETE** | insufficient_pairs — upstream blocked |
| FR-11 relay (live) | 384 | **BLOCKED** | Script exists; GPU node offline |
| Three-tier live benchmark | 385 | **BLOCKED** | Script exists; GPU node offline |
| JitRL constraint memory | 386 | **MISSING** | Session interrupted before implementation |
| Safety KAN classifier | 387 | **MISSING** | Session interrupted before implementation |
| SAVeR live execution | 388 | **BLOCKED** | Script exists; GPU node offline |
| Operational retrospective | 389 | **COMPLETE** | RETRO-019/020/021 opened |

**Milestone-level conclusion:**
2026.06.03 fixed the infrastructure (RETRO-015 CLOSED: LiveGPUGate + session_startup.sh are correct)
but the GPU NODE ITSELF was offline during the conductor session. Zero live results for the FIFTH
consecutive milestone. Three interrupted experiments (378, 386, 387) left implementation gaps.
The session also hit its turn limit partway through, leaving 8 experiments in BLOCKED state.

The single most important operational change for 2026.06.10: **run `nvidia-smi` before starting
the conductor session.** If the GPU node is not online, fix GPU availability FIRST. Do not write
a single line of experiment code until Exp 390 smoke test returns inference_mode='live_gpu'.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: GPU node offline — five consecutive milestones of zero live results (RETRO-019, CRITICAL)

The infrastructure is now correct (RETRO-015 closed). The LiveGPUGate raises immediately if
CARNOT_FORCE_LIVE=1 is not set. session_startup.sh exports the env var. The remaining gap is
OPERATIONAL: the GPU node (2x RTX 3090, CUDA) must be physically powered on and connected
before the conductor session starts.

Eight complete, tested experiment scripts exist and will produce real results the moment the GPU
is online:
- experiment_379_precision_execute.py (74 tests passing)
- experiment_380_humaneval_execute.py (24 tests passing)
- experiment_381_adversarial_execute.py (23 tests passing)
- experiment_382_extraction_execute.py (tests passing)
- experiment_384_relay_live.py (tests passing)
- experiment_385_three_tier_execute.py (tests passing)
- experiment_388_saver_live.py (tests passing)

PRE-FLIGHT PROTOCOL (mandatory): Exp 390 runs first. If it cannot confirm inference_mode='live_gpu',
the conductor MUST stop and escalate rather than writing more blocked experiment code.

### Gap 2: Three missing implementations — CIKANEnergy, JitRL, Safety KAN (RETRO-020, HIGH)

Session interruption in 2026.06.03 left three experiments unimplemented:
- **CIKANEnergy** (RETRO-020): python/carnot/models/cikan_energy.py still contains JSON not Python.
  This is the second consecutive milestone where CIKAN fails. The implementation is well-specified;
  the file just needs to be rewritten from scratch.
- **JitRL Constraint Memory**: The correct Tier 1 algorithm (threshold modulation, not weight
  reweighting). Exp 134 proved reweighting does not work. JitRL's non-parametric memory is the fix.
  arXiv 2601.18510 specifies the algorithm precisely.
- **Safety KAN Classifier**: First Tier B product. Energy-based jailbreak detection, interpretable
  splines, CPU-only with contrastive training on hardcoded examples. 2-3 days of implementation.

These three are CPU-only (no GPU needed) and must complete in Phase 1 regardless of GPU state.

### Gap 3: FR-11 self-learning relay unconfirmed — third consecutive milestone (RETRO-021, HIGH)

The SelfLearningRelay (Exp 361) showed 0.60→0.72 accuracy improvement on SYNTHETIC data. The
live version (Exp 374/384) was blocked by GPU unavailability in two consecutive milestones.
FR-11 (PRD requirement: "Autonomous Self-Learning Loop") cannot be marked CLOSED until
learning_confirmed is produced from live GPU inference.

The target: run 4 batches of 25 live GSM8K questions with Gemma4-E4B-it. Batch 4 accuracy must
exceed batch 1 accuracy (improved=True AND inference_mode="live_gpu") to produce
honest_verdict="learning_confirmed" and close FR-11.

Once FR-11 closes, update _bmad/traceability.md to mark FR-11 COMPLETE. This is the single
most impactful PRD milestone remaining open.

---

## Architecture Snapshot (Post-v35)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CARNOT ARCHITECTURE v35                                  │
│                                                                             │
│  LLM Output                                                                 │
│      │                                                                      │
│  SemanticEnergyScorer ──── (new: logit-space plausibility gate)             │
│      │                                                                      │
│  ConstraintExtractor ──────────────────────────────────────────────────┐   │
│  (CRANE Gate | LLMExtractor | LLMz3Formalizer | CodeExtractor)         │   │
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
│      │         │                                                        │   │
│      ▼         ▼                                                        │   │
│  Self-Learning Relay                                                    │   │
│      Tier 1: PerModelFPTracker (JitRL threshold modulation)             │   │
│      Tier 2: ConstraintTemplateWiring (carry/sign/unit/comparison)      │   │
│      Tier 3: EORM gate (energy-ranked repair candidates)                │   │
│           │                                                             │   │
│      SAVeR Verifier (multi-turn faithfulness audit)                     │   │
│           │                                                                 │
│  Safety KAN (Tier B: jailbreak/unsafe energy classifier)                    │
│                                                                             │
│  Hardware Path:                                                             │
│      CPU: JitRL memory, Ising sampling, CIKAN boundary penalties            │
│      GPU: LLM inference, EORM scorer, JEPA predictor                        │
│      FPGA: Sparse Ising (KV260 — bitfile pending), CIKAN LUT boundaries     │
│      TSU (future): Native thermodynamic sampling (Extropic Z1)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 0: GPU Preflight (Exp 390) — MANDATORY GATE

Run `nvidia-smi` and execute the Exp 353 smoke test. If inference_mode is not 'live_gpu', write a
blocked artifact and STOP. Do not proceed with any Phase 2+ experiments until GPU is confirmed.

Phase 0 produces a preflight report that all subsequent experiments reference. It resolves
RETRO-019 if GPU is confirmed online. If not, RETRO-019 escalates further.

### Phase 1: Missing Implementations (Exps 391-393) — CPU-only, no GPU required

Implement the three experiments interrupted in 2026.06.03. All are CPU-only so they run
regardless of GPU state (the preflight may show GPU offline — these still proceed):

- **Exp 391**: CIKANEnergy — complete Python implementation, RETRO-020 close
- **Exp 392**: JitRL Constraint Memory — threshold modulation implementation
- **Exp 393**: Safety KAN Classifier — first Tier B product, hardcoded training examples

### Phase 2: Live Benchmark Execution (Exps 394-397) — GPU required

Execute the blocked benchmark scripts using live GPU (confirmed by Exp 390). Each execution
experiment has an existing, tested script (from prior milestones) and simply wraps the core
pipeline with LiveGPUGate and runs it:

- **Exp 394**: Precision pipeline (200 GSM8K, 5 variants, 2 models) — Carnot's headline metric
- **Exp 395**: HumanEval code verification (50 problems, CodeExtractor + PBT) — strongest result domain
- **Exp 396**: Adversarial GSM8K (standard + adversarial + repaired) — Carnot's credibility claim
- **Exp 397**: Extraction comparison (regex vs LLMExtractor vs LLMz3Formalizer) — RETRO-016 close

### Phase 3: Self-Learning Confirmation (Exps 398-400) — depends on Phase 2

Use real data from Phase 2 to train and validate the self-learning stack:

- **Exp 398**: Combined EORM+JEPA retrain on live pairs from Exps 394-397
- **Exp 399**: FR-11 relay (4 live batches) — RETRO-021 close attempt
- **Exp 400**: SAVeR live multi-turn verification (5 reasoning chains)

### Phase 4: New Research Experiments (Exps 401-402) — arxiv findings

Implement promising recent papers that directly address Carnot's extraction bottleneck:

- **Exp 401**: Semantic Energy Scorer (arXiv 2508.14496) — logit-space hallucination gating
- **Exp 402**: CRANE Extraction Gate (arXiv 2502.09061) — alternating free + constrained extraction

### Phase 5: Retrospective (Exp 403)

Evaluate all success criteria, close resolved RETRO items, open new ones, update ops docs.

---

## Success Criteria

| Criterion | Target | Experiment |
|-----------|--------|-----------|
| retro_019_resolved | GPU node online at session start | Exp 390 |
| retro_020_closed | CIKANEnergy is Python, not JSON | Exp 391 |
| retro_021_closed | FR-11 learning_confirmed on live GPU | Exp 399 |
| live_gpu_confirmed | Any experiment with inference_mode='live_gpu' | Exps 394-400 |
| precision_result_credible | signed_improvement (live GPU) | Exp 394 |
| humaneval_result_credible | code_verification_positive (live GPU) | Exp 395 |
| adversarial_result_credible | improvement_positive (live GPU) | Exp 396 |
| extraction_winner_known | live_gpu_winner verdict | Exp 397 |
| jitrl_memory_works | threshold_modulation_works=True | Exp 392 |
| safety_kan_works | test_auroc > 0.70 | Exp 393 |
| saver_live_verified | live_verification_active | Exp 400 |
| semantic_energy_viable | auroc > 0.70 vs SinkProbe | Exp 401 |
| crane_extraction_improved | detection_rate > ArithmeticExtractor | Exp 402 |

---

## Dependency Graph

```
Exp 390 (GPU preflight)
├── [if GPU online] → Exp 394 (precision live)
│                  → Exp 395 (HumanEval live)
│                  → Exp 396 (adversarial live)
│                  → Exp 397 (extraction comparison live)
│                  → [Phase 2 complete] → Exp 398 (retrain)
│                                       → Exp 399 (FR-11 relay)
│                                       → Exp 400 (SAVeR live)
├── [always, CPU-only] → Exp 391 (CIKANEnergy)
│                     → Exp 392 (JitRL)
│                     → Exp 393 (Safety KAN)
│                     → Exp 401 (Semantic Energy Scorer)
│                     → Exp 402 (CRANE Extraction Gate)
└── [all phases complete] → Exp 403 (retrospective)
```

---

## Hardware Requirements

| Experiment | GPU Required | Notes |
|-----------|-------------|-------|
| Exp 390 | Yes | GPU preflight — confirms GPU available |
| Exps 391-393 | No | CPU-only implementations |
| Exps 394-400 | Yes | Live inference with Gemma4-E4B-it + Qwen3.5-0.8B |
| Exps 401-402 | Preferred | Semantic Energy needs logits (GPU model); CRANE needs live inference |
| Exp 403 | No | CPU-only retrospective |

**GPU dependency note:** If Exp 390 shows GPU offline, Exps 394-400 will produce blocked artifacts.
Exps 391-393 and 401-402 (CPU-only implementations) proceed regardless. The milestone's
learning and retrospective experiments still produce useful infrastructure and analysis even
without live inference.

**Hardware wishlist reference:**
- KV260 FPGA (arrived): bitfile needed for CIKAN LUT boundary mapping (Exp 391 design)
- AMD XDNA NPU: requires ninja + openblas installation (system admin action)
- 2x RTX 3090 (48GB VRAM, CUDA): PRIMARY GPU — must be powered on before session start

---

## New Papers Incorporated (v35)

| Paper | arXiv | Contribution to v35 |
|-------|-------|---------------------|
| Semantic Energy | 2508.14496 | Exp 401: logit-space hallucination scorer |
| CRANE | 2502.09061 | Exp 402: alternating-grammar extraction gate |
| DSP / Feasibility Channels | 2604.02350 | Future: continuous feasibility in SAVeR |
| Potts MFC | 2602.04200 | Future: sparse constraint embedding for FPGA |
| LLM-QUBO | 2509.00099 | Future: LLMExtractor prompt pattern reference |

---

## What the Experiments Prove

By the end of milestone 2026.06.10, assuming live GPU is online, Carnot will have:

1. **First credible benchmark numbers** — 200 GSM8K, 50 HumanEval, adversarial robustness,
   all with inference_mode='live_gpu'. These are the numbers we can report publicly.

2. **FR-11 CLOSED** — The PRD's autonomous self-learning requirement met on real data.
   Batch 1 → Batch 4 accuracy improvement with live Gemma4-E4B-it.

3. **Extraction winner known** — LLMExtractor vs CRANE vs regex comparison on live IT model
   output. This resolves the single most critical technical question (RETRO-016).

4. **Two new energy products** — Safety KAN (Tier B: jailbreak classifier) and Semantic Energy
   Scorer (Tier A enhancement: logit-space gating).

5. **CIKAN deployed** — The Phase 2 hardware path's Python implementation, testable before
   FPGA bitfile is available.

If GPU is still offline: Phase 1 and Phase 4 implementations (Exps 391-393, 401-402) advance
the codebase, and RETRO-022 must escalate GPU access to the user's immediate attention with
a concrete action plan (physical connection, cloud GPU, etc.).

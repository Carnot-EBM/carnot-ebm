# Carnot Research Roadmap v33: First Live GPU Results, LLMExtractor, and Real-Data Self-Learning

**Created:** 2026-04-15
**Milestone:** 2026.05.27
**Status:** Planned (activates when milestone 2026.05.20 completes)
**Supersedes:** Milestone 2026.05.20 — "Live GPU Unblock, Apple Adversarial GSM8K, and LLM-as-Extractor"
**Informed by:** Exps 351–364, operational retrospective 2026.05.20, v32 carry-forwards
**External inputs (new in v33):**
- CIKAN (2412.03710) — constraint-informed KAN splines with hard constraint priors baked into topology
- Thermodynamic init optimization (2603.24183) — warm-start digital initialization for FPGA/TSU convergence
- RLVR (2506.14245) — RL with verifiable rewards teaches correct reasoning; EORM as reward signal
- StructEval (2505.20139) — comprehensive structured output benchmark for extraction quality

---

## What 2026.05.20 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| Retro-012/013/014 infrastructure close | 351 | **PARTIAL** | No JSON artifact; verify_timeout, kill_zombies checked |
| Live GPU diagnostic | 352 | **COMPLETE** | is_live_capable=True; CARNOT_FORCE_LIVE missing from conductor |
| Live GPU smoke test | 353 | **CI-SAFE ONLY** | inference_mode=ci_skip; CARNOT_FORCE_LIVE never set |
| Apple adversarial GSM8K harness | 354 | **COMPLETE** | 63 tests, harness_ready=True |
| Apple adversarial GSM8K execution | 355 | **BLOCKED** | honest_verdict=blocked_simulated; live pending |
| LLMExtractor implementation | 356 | **SKIPPED** | Never implemented — RETRO-013 |
| LLMz3Formalizer | 357 | **COMPLETE** | 58 tests, ci_stub mode; live pending |
| Extraction benchmark harness | 358 | **COMPLETE (CI)** | 33 tests, no result JSON (RETRO-014) |
| EORM real-data retrain | 359 | **SYNTHETIC ONLY** | 5 real pairs, auc_improvement=0.000 |
| Three-tier pipeline benchmark | 360 | **CI SYNTHETIC** | 54 tests, inference_mode=cpu_synthetic |
| Self-learning relay (FR-11) | 361 | **SYNTHETIC** | 0.60→0.72, honest_verdict=synthetic_only |
| SAVeR multi-turn wrapper | 362 | **COMPLETE (CI)** | 31 tests, CI-safe; live pending |
| Operational retrospective 2026.05.20 | 363 | **COMPLETE** | RETRO-012/013/014 identified |
| ModelServer + TensorRT + DualGPU wiring | 364 | **COMPLETE** | All harnesses wired for hardware-accelerated testing |

**Milestone-level conclusion:**
2026.05.20 wired the full infrastructure (ModelServer, TensorRT, DualGPU assignment, smoke test,
live GPU diagnostic) and built every module needed for live experiments. SAVeR multi-turn wrapper
is complete. Self-learning relay shows improvement (synthetic). The three-tier pipeline is assembled.

However, the milestone failed on every success criterion involving live GPU: CARNOT_FORCE_LIVE was
never set by the conductor for the third consecutive milestone. The diagnostic confirmed hardware is
ready (is_live_capable=True). RETRO-012 is a one-line fix that unblocks all live benchmarks. Until
this is fixed, every benchmark number is a simulation artifact.

Additionally, the LLMExtractor (Exp 356) was skipped entirely — the extraction bottleneck that caused
zero violations on Gemma4-E4B-it remains unresolved. Self-learning trained only on synthetic data.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Live GPU inference — the conductor never sets CARNOT_FORCE_LIVE (RETRO-012)

Three consecutive milestones (2026.05.06, 2026.05.13, 2026.05.20) produced zero live GPU results.
Exp 352 confirmed: `is_live_capable=True` — both RTX 3090s are healthy, CUDA is visible, models are
loadable. The failure is not hardware. It is the conductor not propagating `CARNOT_FORCE_LIVE=1` into
the subprocess environment for GPU-tagged tasks.

RETRO-012 is the highest-ROI action in the entire history of this project (est. 12% savings per
milestone by eliminating simulated retries). It is a one-line fix in `scripts/research_conductor.py`
... but wait, we cannot modify `scripts/research_conductor.py` from an experiment prompt. The fix
must be delivered in a wrapper script or an environment file that the conductor sources.

**Resolution path:** Exp 365 creates a `scripts/conductor_gpu_env.sh` that exports
`CARNOT_FORCE_LIVE=1` and is sourced before GPU-tagged experiments. Documents the RETRO-012 fix
so it can be applied to the conductor manually.

**This milestone's priority #1: Close RETRO-012/013/014 with documented fixes. Verify live GPU
works via smoke test before any benchmark experiment runs.**

### Gap 2: LLMExtractor (Exp 356) still skipped — constraint extraction broken for IT models

ArithmeticExtractor finds zero violations on Gemma4-E4B-it because instruction-tuned models do not
write equations in `a + b = c` regex format. Exp 356 was planned for milestone 2026.05.20 but never
ran. This means the extraction benchmark (Exp 358) cannot produce a live honest_verdict — it only
has the ArithmeticExtractor and LLMz3Formalizer paths, neither of which works on IT format.

LLMExtractor uses a second small LLM call (same Qwen3.5-0.8B used for Carnot's own pipeline) to
extract verifiable claims from the response in a structured format. The claims become constraint
inputs to the existing Ising/Z3 verification stack. This is the research-program.md Goal #1b
"LLM-as-extractor" approach.

**This milestone's priority #2: Implement LLMExtractor as a first-class extractor. Wire it into
the extraction benchmark. Run live comparison: LLMExtractor vs ArithmeticExtractor vs LLMz3Formalizer
on Gemma4-E4B-it output. Measure violation detection rate on questions with known wrong answers.**

### Gap 3: Self-learning validated only on synthetic data — no learning_confirmed verdict

The self-learning relay (Exp 361) showed accuracy improvement (0.60→0.72) but `honest_verdict=synthetic_only`.
EORM trained on 5 real pairs (AUC 0.500 — unchanged). JEPA retrained on 60 synthetic pairs.
The three-tier pipeline ran in `cpu_synthetic` mode. Without live GPU data:
- EORM cannot distinguish correct from incorrect CoT responses
- JEPA cannot learn which partial responses predict violations
- The self-learning relay cannot achieve `learning_confirmed`

This is a downstream consequence of Gap 1. Once live benchmarks run (priority #1), real
(question, response, correctness) pairs become available for training. Exps 371-374 re-run the
self-learning stack on real data.

**This milestone's priority #3: With live GPU benchmarks providing real data, retrain EORM and
JEPA on real pairs. Measure genuine AUC improvement. Run the self-learning relay on live GPU
to achieve the first `learning_confirmed` verdict.**

---

## System Architecture (No Changes)

The architecture from milestone 2026.05.20 is complete. This milestone focuses on running it
on real data for the first time.

```
Extraction tier (IT models):
  LLM response → LLMExtractor (small LLM call) → structured claims → ArithmeticExtractor
               → LLMz3Formalizer (Z3 assertions) → Z3 verification
               → NL2Z3Extractor → NL constraint terms

Three-tier verification cascade (Exp 360):
  LLM response → SinkProbe (attention sinks, fast) → skip if low uncertainty
               → EORM (energy reward model, medium) → skip if low energy
               → Ising (full constraint check, slow) → verify/repair

Self-learning pipeline (Exp 361):
  Per query:
    Tier 1: PerModelFPTracker.update(was_fp, was_tp)         # CPU counter, <1μs
    Tier 2: CaseMemoryTemplateWiring.on_violation_recorded() # CPU+memory, <1ms
    Tier 3: EORM.energy(response, question)                  # GPU inference, <10ms
  Cross-session: SessionMemory saves/loads all three tiers

Instrumentation (Exp 364):
  All benchmarks use ModelServer + TensorRT cache + DualGPURunner
  GPU 0: Gemma4-E4B-it; GPU 1: Qwen3.5-0.8B
  Explicit CARNOT_FORCE_LIVE=1 required for live results
```

---

## Dependency Graph

```
Phase 1: RETRO closes (no deps — always safe to run)
  Exp 365: RETRO-012/013/014 close + conductor env fix

Phase 2: LLMExtractor (depends on nothing, just coding)
  Exp 366: LLMExtractor implementation
  Exp 367: Live extraction benchmark (Exp 358 re-run)  ← needs Exp 366

Phase 3: Live GPU benchmarks (depends on Phase 1 + CARNOT_FORCE_LIVE in env)
  Exp 368: Live precision pipeline benchmark
  Exp 369: Live HumanEval code verification
  Exp 370: Live adversarial GSM8K

Phase 4: Real-data self-learning (depends on Phase 3 results)
  Exp 371: EORM real-data retrain   ← needs Exps 368/369/370 result JSONs
  Exp 372: JEPA retrain on real pairs  ← needs Exps 368/369 result JSONs
  Exp 373: Three-tier pipeline on live GPU  ← needs Exps 368 (attention matrices)

Phase 5: New capability + FR-11 + retro
  Exp 374: Self-learning relay on live GPU  ← needs Exps 371/372/373
  Exp 375: CIKAN constraint-informed KAN energy tier (new capability, no deps)
  Exp 376: Operational retrospective
```

---

## Hardware Requirements

| Experiment | Hardware | Why |
|------------|----------|-----|
| Exp 365 | CPU | Documentation + env script |
| Exp 366 | CPU + GPU optional | LLMExtractor implementation (CI-safe) |
| Exp 367 | 2x RTX 3090 | Live extraction comparison |
| Exp 368 | 2x RTX 3090 | Live precision benchmark (200 GSM8K questions) |
| Exp 369 | 2x RTX 3090 | Live HumanEval (50 problems) |
| Exp 370 | 2x RTX 3090 | Live adversarial GSM8K (50 questions × 3 conditions) |
| Exp 371 | 1x RTX 3090 | EORM training on real pairs |
| Exp 372 | 1x RTX 3090 | JEPA training on real violation pairs |
| Exp 373 | 2x RTX 3090 | Three-tier pipeline with real attention matrices |
| Exp 374 | 2x RTX 3090 | Self-learning relay (4 batches of 25 live questions) |
| Exp 375 | CPU | CIKAN energy tier (no LLM inference needed) |
| Exp 376 | CPU | Retrospective (data analysis) |

**FPGA (KV260):** Not required this milestone. Bitfile still pending (see research-hardware-wishlist.md).
**AMD XDNA NPU:** Still blocked by ninja + openblas prerequisite. Human install required.

---

## Success Criteria

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| live_gpu_confirmed | **True** | At least one experiment with inference_mode="live_gpu" |
| llm_extractor_beats_regex | **True** | Exp 367: LLMExtractor detection_rate > ArithmeticExtractor |
| adversarial_result_credible | **True** | Exp 370: honest_verdict=improvement_positive |
| eorm_retrained_on_real | **True** | Exp 371: retrain_mode=real_data, auc_improvement > 0 |
| self_learning_confirmed | **True** | Exp 374: honest_verdict=learning_confirmed |
| retro_012_closed | **True** | conductor_gpu_env.sh created and documented |
| all_result_jsons_present | **True** | All 12 experiments produce results/*.json |
| cikan_implemented | **True** | Exp 375: python/carnot/models/cikan_energy.py exists and tests pass |

---

## Phase Descriptions

### Phase 1: Retro Closures + Conductor Environment Fix (Exp 365)

RETRO-012 is the single highest-impact action available. For the third consecutive milestone, live
GPU inference was blocked because CARNOT_FORCE_LIVE=1 was never in the conductor subprocess
environment. The fix: create `scripts/conductor_gpu_env.sh` that exports the variable, and document
how the human/conductor operator applies it before launching GPU-tagged experiments. Also close
RETRO-014 by enforcing that every experiment script ends with an explicit `artifact.write(...)` call.

RETRO-013 is addressed not in this phase but in Phase 2 (LLMExtractor implementation).

### Phase 2: LLMExtractor — Fix Constraint Extraction for IT Models (Exps 366-367)

The ArithmeticExtractor's regex (`a + b = c`) finds zero violations on Gemma4-E4B-it. Instruction-tuned
models use natural language ("the total is 47") not equations. LLMExtractor uses a second LLM call
to extract verifiable claims:

```
LLM response (IT format) → LLMExtractor prompt → small LLM → structured claims JSON
  [{"lhs": "47", "rhs": "28+19", "op": "+", "claim": "47 equals 28 plus 19"}]
→ ArithmeticExtractor (structured claims) → IsingConstraint
```

This is the MathAgent "Legislator" pattern (arXiv 2604.11188): a second model generates the
constraint representation that the first model's output is verified against. The structured claim
format enables Z3 verification (LLMz3Formalizer path) in addition to Ising.

Exp 367 runs the live extraction comparison. Questions where the model's answer is wrong should
show high violation detection rate with LLMExtractor (if extraction works) but zero with regex.

### Phase 3: Live GPU Benchmarks — First Real Numbers (Exps 368-370)

Three experiments re-run their simulation-only predecessors with CARNOT_FORCE_LIVE=1 enforced:

**Exp 368** (Live Precision Pipeline Benchmark): Re-runs Exp 340's 5-variant × 2-model × 200-question
GSM8K benchmark. This is the experiment that has been "pending live GPU" since milestone 2026.05.06.
With Exp 364's ModelServer + DualGPURunner wiring, the infrastructure is ready. This produces the
first live headline result for the precision stack.

**Exp 369** (Live HumanEval Code Verification): Re-runs Exp 341. Code verification is our strongest
result category because it uses execution-based verification (not regex). Exp 226 showed +3.0pp on
Gemma4-E4B-it. Exp 369 gets a live result with the full precision stack.

**Exp 370** (Live Apple Adversarial GSM8K): Re-runs Exp 355. This is Carnot's headline credibility
experiment. If verify-repair maintains accuracy while baseline drops on adversarial variants, that
is the result that validates the entire verification approach.

### Phase 4: Real-Data Self-Learning (Exps 371-373)

With Phase 3 providing real (question, response, is_correct) pairs:

**Exp 371** (EORM Real-Data Retrain): Re-runs Exp 359 with real pairs from Exps 368-370. The EORM
needs at least 50 real pairs with mix of correct/incorrect responses per question. With 200+50 GSM8K
and 50 HumanEval live results, this threshold is easily exceeded. Target: AUC-ROC improvement from
0.500 baseline.

**Exp 372** (JEPA Real-Data Retrain): Re-runs Exp 347 with real violation pairs. Splits each live
response at prefix_fraction=0.5, pairs (partial, full, has_violation). Trains JEPA predictor on real
violation patterns. Target: AUC improvement and better skip-rate at same FNR on real inputs.

**Exp 373** (Three-Tier Pipeline on Live GPU): Re-runs Exp 360 with real attention matrices from live
inference (requires `output_attentions=True` in model call). Measures real skip_rate, fn_rate, and
throughput improvement vs Ising-alone on live responses.

### Phase 5: New Capability + FR-11 Live + Retro (Exps 374-376)

**Exp 374** (Self-Learning Relay on Live GPU): Re-runs Exp 361 with live inference. This is the FR-11
mandatory milestone goal: the first `learning_confirmed` verdict requires `inference_mode=live_gpu`
AND `improved=True`. With real data from Phase 3 and retrained EORM/JEPA from Phase 4, this should
show genuine learning improvement rather than synthetic progression.

**Exp 375** (CIKAN Constraint-Informed KAN Energy Tier): New capability from arXiv 2412.03710.
Implements `CIKANEnergy` as a subclass of `KANEnergy` where spline activations are seeded with
known constraint boundaries. Carry-check constraints have infinite energy at the boundary (carries
that don't propagate). Range-check constraints have infinite energy outside valid ranges. Compare
CIKAN vs standard KAN on constraint satisfaction tasks. This is a hardware-path capability: FPGA
spline LUTs can encode the constraint boundaries as hard-wired saturations.

**Exp 376** (Operational Retrospective): Measures whether RETRO-012 fix actually reduced wall time.
If live GPU now works, batch size 8 produces ~8x throughput vs one-at-a-time simulated responses.
This milestone should be dramatically faster than 2026.05.20's 33.3 min/exp average.

---

## New Papers Added to research-references.md

| Paper | ID | Why Added |
|-------|----|-----------|
| CIKAN: Constraint Informed KAN Networks | 2412.03710 | Hard constraint priors in splines → KAN energy tier extension |
| Digitally Optimized Initializations for Thermodynamic Computing | 2603.24183 | Warm-start for FPGA/TSU convergence |
| RLVR — Reinforcement Learning with Verifiable Rewards | 2506.14245 | EORM-as-reward for constraint extraction fine-tuning |
| StructEval: Structured Output Benchmark | 2505.20139 | Benchmarking LLMExtractor output quality |

---

## What This Milestone Does NOT Attempt

- **KV260 FPGA bring-up:** Still blocked by missing bitfile. Requires human-built bitfile.
- **AMD XDNA NPU:** Still blocked by ninja + openblas (4th consecutive milestone). Human install required.
- **Full-scale 1319-question GSM8K:** Too large for single session. 200-question live benchmark is sufficient for Phase 1 live credibility.
- **RLVR fine-tuning of LLMExtractor:** Interesting but requires real violation pairs first. Deferred to milestone after this one.
- **D-Wave QPU integration:** Free tier (1 min/month) is too small for benchmarking. Deferred.

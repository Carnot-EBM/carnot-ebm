# Carnot Research Roadmap v32: Live GPU Unblock, Apple Adversarial GSM8K, and LLM-as-Extractor

**Created:** 2026-04-15
**Milestone:** 2026.05.20
**Status:** Planned (activates when milestone 2026.05.13 completes)
**Supersedes:** Milestone 2026.05.13 — "Live E2E Precision Pipeline, Constraint Addition from Memory, and EORM Predictive Verification"
**Informed by:** Exps 338-350, operational retrospective 2026.05.13, v31 carry-forwards
**External inputs (new in v32):**
- ARM-EBM Bijection (2512.15605) — formal proof that ARMs are secretly EBMs; grounds spilled energy
- SAVeR (2604.08401) — self-auditing multi-turn verification with constraint-guided repair-before-commit
- MathAgent (2604.11188) — legislator-executor constraint graph generation as LLM extraction blueprint
- T-SKM-Net (2512.10461) — trainable neural constraint satisfaction via algebraic projection
- Online CoT verifier learnability bounds (2603.03538) — soundness/completeness trade-offs for online verification

---

## What 2026.05.13 Proved

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| Host prereqs registry + DualGPURunner default | 338 | **COMPLETE** | RETRO-004 + RETRO-006 closed |
| Pre-session startup health check | 339 | **COMPLETE** | RETRO-007 + RETRO-008 closed |
| Live full precision pipeline benchmark | 340 | **SIMULATED** | inference_mode=simulated; live GPU unused |
| Live HumanEval code verification | 341 | **SIMULATED** | inference_mode=simulated; live GPU unused |
| ConstraintTemplateLibrary — Tier 2 constraint addition | 343 | **COMPLETE** | 4 built-in templates; positive improvement_delta |
| CaseMemory → ConstraintTemplateLibrary wiring | 344 | **COMPLETE** | constraint addition shows positive delta vs Exp 134 reweighting |
| SessionMemory — cross-session persistence | 345 | **COMPLETE** | save/load round-trip verified; carnot.session_memory.v1 |
| EORM CoT energy reward model | 346 | **SIMULATED** | inference_mode=simulated; trained on synthetic pairs |
| JEPA real-data retrain | 347 | **SIMULATED** | Exp 340 had no "responses" key; fallback to synthetic |
| SinkProbe attention-sink pre-filter | 348 | **COMPLETE** | 60% skip rate, FNR=0%, TNR=100% on synthetic |
| KV260 FPGA open-source bitfile synthesis | 349 | **COMPLETE/PARTIAL** | yosys synthesis attempted; bitfile status in results |
| Operational retrospective 2026.05.13 | 350 | **COMPLETE** | 399 experiments; RETRO-009/010/011 opened |

**Milestone-level conclusion:**
2026.05.13 delivered the full Tier 2 self-learning stack (ConstraintTemplateLibrary, CaseMemory wiring,
SessionMemory), the SinkProbe pre-filter, and infrastructure closes for RETRO-004/006/007/008.

However, the fundamental credibility problem persists: **all four live-GPU experiments ran in simulated
mode.** Both RTX 3090s were idle for the entire milestone. EORM and JEPA trained on synthetic data.
RETRO-003 remains open for a second consecutive milestone. Without live GPU inference, every benchmark
number is unverified.

Additionally, constraint extraction is still broken for instruction-tuned models: ArithmeticExtractor
found zero violations on Gemma4-E4B-it (0/20 questions, including 4 wrong answers). The LLM-as-extractor
approach (research-program.md Goal #1b) has not been implemented.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Live GPU inference never works — all benchmarks are simulated artifacts

Two consecutive milestones (2026.05.06 and 2026.05.13) produced zero live GPU inference results.
The RTX 3090s sit idle while the conductor falls through to simulated fallback. This is not a hardware
problem — it is a software path problem. Something in the `CARNOT_FORCE_LIVE=1` + model-loading path
triggers simulated mode rather than raising an explicit error.

Root cause is unknown because no experiment has traced the failure path. Exp 340 shows
`inference_mode=simulated` in results but does not report WHY the live path failed. The conductor
retries and gets the same result. This wastes experiment slots on simulated benchmarks that cannot
produce credible numbers.

**This milestone's priority #1: Trace the live GPU inference failure path, fix the root cause,
and verify with a live smoke test before any benchmark experiments run. RETRO-003 must also close
(timeout wiring) — both are prerequisites for every subsequent experiment.**

### Gap 2: Constraint extraction broken for IT models — no credible verification results

ArithmeticExtractor finds zero violations on Gemma4-E4B-it. The regex pattern (`a + b = c`) does not
match instruction-tuned output formats. This means verify-repair cannot fix errors on the models users
actually deploy. The pipeline infrastructure is solid but the extraction layer is the bottleneck.

Two approaches are available (research-program.md Goal #1):
1. **LLM-as-extractor**: A second small LLM call extracts verifiable claims from the response in a
   structured format that ArithmeticExtractor can parse. More flexible, handles any response format.
2. **LLM-guided Z3** (arXiv 2601.04675): An LLM rewrites ambiguous arithmetic as explicit Z3 assertions,
   then Z3 verifies. 80% improvement in Z3 success rate on previously-failing inputs.

Both must be tested on live instruction-tuned models (Gemma4-E4B-it). No more simulated extraction.

**This milestone's priority #2: Implement LLMExtractor and LLM-guided Z3 formalization, benchmark
both against ArithmeticExtractor on live IT model output. Measure violation detection rate directly.**

### Gap 3: Apple adversarial GSM8K — the credibility experiment — still untested

Research-references.md labels the Apple adversarial GSM8K test as "Carnot's most compelling result"
and "the credibility experiment." The adversarial variant (arXiv 2410.05229) adds irrelevant sentences
to identical math problems, causing models to drop up to 65% accuracy. Carnot's arithmetic verifier
should be immune because it extracts and verifies arithmetic regardless of irrelevant context.

This experiment has been deferred for multiple milestones because live GPU inference was broken.
With Gap 1 fixed, this becomes the primary benchmark target for the milestone.

**This milestone's priority #3: Run the Apple adversarial GSM8K benchmark on live GPU with
Gemma4-E4B-it. Measure: (a) accuracy drop on adversarial variants, (b) verify-repair recovery rate.
If verify-repair maintains accuracy on adversarial inputs where the model fails, that is Carnot's
headline result.**

---

## Architecture: v32 Additions

```
[Input Query]
    │
    ├─[SinkProbe]──────────────────── attention sink signal   (EXISTING: Exp 348)
    │  arXiv 2604.10697               fast pre-filter; if sink score < threshold, skip Ising
    │
    ▼
[LLM Generation — RTX 3090 x2, CARNOT_FORCE_LIVE=1]         (UNBLOCKED: Exps 352-353)
    │   ← live GPU inference finally working
    │
    ├─[LLMExtractor]──────────────── structured claims        (NEW: Exp 356)
    │   second LLM call: "what arithmetic claims does this make?"
    │   output: JSON list of (lhs, op, rhs, result) triples
    │       ↓
    │  [LLM-guided Z3]───────────── explicit Z3 assertions    (NEW: Exp 357)
    │   arXiv 2601.04675             LLM rewrites claims as z3.add() statements
    │   80% improvement in Z3 success rate on ambiguous inputs
    │
    ├─[EORM Energy Ranker]──────────── CoT ranking score      (RETRAINED: Exp 359)
    │  arXiv 2505.14999               retrained on real (CoT, correctness) pairs
    │                                 from Exps 340/341/355
    │
    ├─[JEPA Fast-Path Gate]──────────── violation predicted?  (EXISTING: Exp 347)
    │       ↓ if high energy
    │
    ├─[CoTCircuitVerifier] ──────────── structural CoT graph  (EXISTING: Exp 336)
    │  arXiv 2510.09312               broken circuit → flag
    │
    ▼
[Ising Verification — CPU fast path]
    │
    ├─[ConfidenceWeightedRepair] ──── score × confidence      (EXISTING: Exp 332)
    ├─[ModelAdaptiveThresholds] ──── per-model FP tracking    (EXISTING: Exp 333)
    └─[VERGE Iterative Loop] ──────── targeted step repair    (EXISTING: Exp 334)
    │
    ▼
[Three-Tier Fast/Medium/Slow Pipeline]                        (BENCHMARKED: Exp 360)
    ├─[SinkProbe] → fast gate (0ms)
    ├─[EORM] → medium gate (real-data trained)
    └─[Ising] → slow gate (accurate)
    │
    ▼
[Tier 1: Online constraint weight updates]                    (EXISTING)
[Tier 2: CaseMemory → ConstraintTemplateLibrary]             (EXISTING: Exps 343-345)
[Tier 3: EORM + JEPA gate on real data]                      (UPGRADED: Exps 359-360)
[Tier 1+2+3 Relay — real models, real data]                  (NEW: Exp 361)
    │   continuous self-learning experiment — satisfies FR-11
    │
    ▼
[Multi-Turn Agentic Verification — SAVeR]                     (NEW: Exp 362)
    │  arXiv 2604.08401
    │  constraint state persists across agent steps
    └─[ConstraintStateMachine] ── propagation across turns
    │
    ▼
[Hardware backends]
    ├─[FpgaBackend] ─── KV260 bitfile bring-up                (CARRY-FORWARD)
    └─[AMD XDNA NPU] ── blocked (ninja + openblas — human install required)
```

---

## Phase Design

### Phase 1: Critical Infrastructure (Exps 351-353)

Close RETRO-003/005/009/010/011. Diagnose and fix the live GPU inference failure path.
Verify live GPU with a smoke test before any benchmark work. These are prerequisites
for every subsequent experiment.

**RETRO-003** (highest priority, must close before any benchmark): wire
`run_experiment_with_timeout.sh` as mandatory wrapper in conductor invocation. Zero
implementation work required — it is already written.

**RETRO-005**: wire `gpu_monitor.py --kill-zombies` into conductor inter-experiment cleanup.

**RETRO-009/010/011**: add live GPU smoke test at session start (RETRO-009); document
presplit guidance for complex experiments >30 min estimated (RETRO-010); add batch doc
reconciliation every 5 experiments to conductor runner (RETRO-011).

### Phase 2: Apple Adversarial GSM8K (Exps 354-355)

Implement the Apple adversarial GSM8K harness (script + tests, no GPU) then execute
on live GPU. Two-part split per research-program.md "Large Benchmark Experiments" rule.

Target: Gemma4-E4B-it on 100 standard + 100 adversarial GSM8K questions.
Measure: accuracy drop (standard → adversarial) and verify-repair recovery on adversarial.
If verify-repair is immune to irrelevant-sentence perturbation, report as headline result.

### Phase 3: LLM-as-Extractor (Exps 356-358)

Implement two new extraction approaches (both must run on live IT model output):
1. **LLMExtractor**: small LLM call that converts free-form reasoning into parseable claim triples.
2. **LLM-guided Z3**: LLM rewrites ambiguous arithmetic into explicit Z3 assertion code.

Both informed by MathAgent's constraint graph generation (arXiv 2604.11188): the LLMExtractor
output format is a constraint graph (variable bindings + arithmetic assertions), not just a
list of regex patterns. This format maps directly to Ising coupling matrices.

Benchmark experiment (Exp 358) compares ArithmeticExtractor vs LLMExtractor vs LLM-guided Z3
on 50 live questions from Gemma4-E4B-it. Primary metric: violation detection rate on known-wrong
answers (the Gemma4 questions where it answers incorrectly).

### Phase 4: Real-Data EORM + Three-Tier Pipeline (Exps 359-360)

EORM and JEPA need to retrain on real (question, CoT, correctness) pairs — these are now
available from Exps 340/341/355 (if live GPU works). EORM retrain on 200+ real pairs,
compare AUC-ROC before/after. Three-tier benchmark measures throughput and accuracy for
the combined SinkProbe → EORM → Ising chain vs Ising-alone baseline.

### Phase 5: Self-Learning Relay + Multi-Turn (Exps 361-363)

**Exp 361** is the mandatory continuous self-learning experiment (research-program.md requirement).
Full Tier 1+2+3 relay: run 100 questions, after each batch update constraint weights (Tier 1),
accumulate patterns into ConstraintTemplateLibrary (Tier 2), evaluate JEPA gate accuracy (Tier 3).
Measure whether accuracy improves across batches.

**Exp 362** implements SAVeR-style multi-turn agentic verification (arXiv 2604.08401): a two-turn
agent loop where the constraint state from step N gates action commitment at step N+1.

**Exp 363** is the operational retrospective.

---

## Dependency Graph

```
[Exp 351: RETRO-003/005 closes]
    │
    └──► [Exp 352: Live GPU diagnostic]
              │
              └──► [Exp 353: Live GPU smoke test]
                        │
              ┌─────────┴────────────────────────────┐
              │                                       │
    [Exp 354: Adversarial dataset]         [Exp 356: LLMExtractor impl]
              │                                       │
    [Exp 355: Adversarial benchmark]       [Exp 357: LLM-guided Z3]
              │                                       │
              └──────────────┬────────────────────────┘
                             │
                   [Exp 358: Extraction benchmark]
                             │
                   [Exp 359: EORM retrain on real data]
                             │
                   [Exp 360: Three-tier pipeline benchmark]
                             │
                   [Exp 361: Tier 1+2+3 self-learning relay]
                             │
                   [Exp 362: SAVeR multi-turn verification]
                             │
                   [Exp 363: Operational retrospective]
```

---

## Open RETRO Items Coming In

| Item | Source | Status | Action |
|------|---------|--------|--------|
| RETRO-003 | 2026.05.06 retro | OPEN (2nd milestone) | Wire timeout.sh — Exp 351 |
| RETRO-005 | 2026.05.06 retro | PARTIAL | Wire kill-zombies — Exp 351 |
| RETRO-009 | 2026.05.13 retro | NEW | Live GPU smoke test — Exp 351 |
| RETRO-010 | 2026.05.13 retro | NEW | Presplit complex experiments — Exp 351 |
| RETRO-011 | 2026.05.13 retro | NEW | Batch doc reconciliation — Exp 351 |
| AMD XDNA NPU | Exps 292/303/314/335 | BLOCKED | Human install: ninja + openblas |
| KV260 bitfile | Exp 313 | BLOCKED | Set CARNOT_KV260_BITFILE after synthesis |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|------------|----------|-------|
| Exp 353: live smoke test | 2x RTX 3090 | CARNOT_FORCE_LIVE=1 required |
| Exp 355: adversarial benchmark | 2x RTX 3090 | DualGPURunner for 2 models |
| Exp 358: extraction benchmark | 2x RTX 3090 | LLM inference needed |
| Exp 359: EORM retrain | 1x RTX 3090 | 200+ real pairs, GPU training |
| Exp 360: three-tier benchmark | 2x RTX 3090 | Full pipeline benchmark |
| Exp 361: self-learning relay | 2x RTX 3090 | 100 questions real inference |
| Exp 362: SAVeR multi-turn | 2x RTX 3090 | Two-turn agent loop |

**RETRO-009 enforcement**: Exp 351 must wire a live GPU smoke test into the session startup
procedure. Subsequent experiments should fail fast (blocked artifact) rather than running in
simulated mode silently. The smoke test is: load Gemma4-E4B-it, run 2 questions, verify
inference_mode=live_gpu in output.

---

## Success Criteria for This Milestone

1. **Live GPU confirmed working**: at least one experiment reports `inference_mode=live_gpu`
2. **Apple adversarial result**: Exp 355 reports accuracy drop + verify-repair recovery,
   signed improvement > 0 on adversarial variant
3. **LLMExtractor beats regex**: Exp 358 shows LLMExtractor detects violations on
   instruction-tuned model output where ArithmeticExtractor found zero
4. **EORM trained on real data**: Exp 359 shows AUC-ROC improvement after retrain on
   real (CoT, correctness) pairs vs synthetic-only baseline
5. **Self-learning relay runs**: Exp 361 runs end-to-end with real models, reports
   constraint weight updates across batches
6. **RETRO-003 closed**: conductor timeout wiring confirmed in ops/conductor-log.md

# Carnot Research Roadmap v23: Formal Claim Verification, Process Integrity, and Predictive Self-Learning

**Created:** 2026-04-13
**Milestone:** 2026.04.18
**Status:** Planned (activates when milestone 2026.04.17 completes)
**Supersedes:** Milestone 2026.04.17 - "Calibrated Verification, Spec-Grounded Code Repair, and Self-Learning"
**Informed by:** Exp 235, Exp 238, Exp 241, Exp 242, Exp 243, VERIFY-038, VERIFY-039, VERIFY-040
**External inputs:** VERGE (2601.20055), ReLoop (2602.15983), Scalable Connectivity for Ising Machines (2503.01177), Decomposing Large-Scale Ising Problems on FPGAs (2602.15985), OpenReview 2026 process-verification scan, Extropic writing / XTR-0 positioning, Kona architecture notes

## What 2026.04.17 Proved

| Approach | Experiments | Finding |
|----------|-------------|---------|
| Calibrated semantic verification | 232-235 | Claim isolation and abstention improved observability and cut Qwen false positives (**7 -> 4**), but verify-only remained harmful on both target models and Gemma false positives worsened (**23 -> 26**). |
| Explicit spec grounding for code | 236-238 | Spec-aware verification now composes cleanly with official tests and PBT, but the identical-stack live comparison is still only a **30-case** paired cohort and the lift remains modest and model-dependent. |
| Case-based self-learning | 239-241 | Case memory and policy compilation raised retrieval precision from Exp 223's **5.8%** to roughly **40%+**, but the milestone's primary held-out gain criterion still failed: all replay strategies stayed flat at **34.48%** held-out success. |
| Hardware-adjacent reranking | 242-243 | The KV260 path is still blocked at bring-up time, and sampler-backed reranking improved replay latency economics but produced **0.0** quality lift on saved repair candidates. |

**The milestone-level conclusion:** Carnot has now exhausted the "better calibration, richer retrieval, same verifier" path. To move toward the PRD vision, the next milestone must upgrade the verifier substrate itself, check process integrity rather than only outcomes, and turn self-learning into proactive behavior changes instead of better replay bookkeeping.

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Live verifiable reasoning is still too heuristic on real instruction-tuned outputs

The semantic path can now parse and score live Qwen3.5-0.8B and Gemma4-E4B-it responses, but it still relies too heavily on scalar evidence aggregation over claims that were never fully formalized. Exp 235 showed that better calibration alone does not earn a safe verify-only path. Carnot still needs a stronger bridge from natural-language reasoning traces to deterministic checks.

### Gap 2: Outcome correctness still hides invalid reasoning and repair processes

Spec-aware code verification is the strongest live lane, but it still mainly judges end states: official tests, PBT, and explicit specs. The current stack does not yet explicitly separate "correct answer by a valid process" from "correct answer for the wrong reason." The same problem applies to semantic reasoning traces. This is now the main credibility gap for small-model verification.

### Gap 3: Self-learning improves retrieval quality, not future task success

Exp 241 closed the loop on the current Tier 1 / Tier 2 design: better keys and compiled policies alone do not create held-out gains. To satisfy FR-11, Carnot needs a self-learning path that adds new constraints, predicts future violations early, and preserves a hardware acceleration story for repeated inference-speed updates.

## Promising 2025-2026 Inputs Adopted in v23

- **Formal claim routing over monolithic judging:** VERGE argues for typed symbolic claims, solver routing, and minimal correction subsets instead of coarse holistic reasoning scores. This directly motivates Exp 244-247.
- **Process verification for small models:** the 2026 OpenReview process-verification thread reinforces that small models can land on correct answers with invalid intermediate reasoning. This motivates Exp 248-251.
- **Behavioral verification instead of single-trace trust:** ReLoop shows that perturbation-based behavioral checks expose brittle but superficially correct solutions. This informs the next code-verification benchmark and process corpus.
- **Hardware-friendly sparse connectivity, not dense wishful thinking:** recent FPGA Ising papers emphasize sparse copy-node compilation and hardware-aware decomposition. That supports shaping verifier workloads for sparse acceleration rather than spending another milestone on blocked overlay plumbing alone.
- **Extropic and Kona still validate the direction, not the next bottleneck:** both continue to support Carnot's validity-layer architecture, but the immediate blocker is verifier quality on real traces, not a lack of yet another backend abstraction.

## v23 Hypothesis

If Carnot replaces scalar semantic scoring with solver-routed formal claims, adds explicit process-integrity checks to both reasoning and code-repair paths, and upgrades self-learning from retrieval-only reuse to constraint addition plus predictive gating, it should be able to produce:

1. its first non-harmful live verify-only reasoning result on at least one target small model,
2. a stronger cross-model code-verification signal on Qwen3.5-0.8B and Gemma4-E4B-it, and
3. its first honest held-out self-learning gain under a zero-extra-false-positive budget.

## v23 Architecture: Solver-Routed Claims Feeding Process Verification and Predictive Learning

```
Prompt / Benchmark Item
        |
        v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Output Contract Policy                                                      │
│  existing terse / minimal JSON / grammar-gated routing                      │
│  goal: request only the structure needed for formal claim or process checks │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                |
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Typed Reasoning + Trace Emission                                             │
│  existing typed reasoning IR, structured reasoning, code spec surfaces       │
│  plus new formal claim corpus + process-integrity corpus                     │
└───────────────┬───────────────────────────────┬──────────────────────────────┘
                |                               |
                v                               v
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│ Formal Claim Verifier        │   │ Process Verifier                         │
│  Exp 244-247                 │   │  Exp 248-251                            │
│  normalize claims            │   │  right-for-wrong-reasons checks         │
│  route to arithmetic / SMT   │   │  step integrity and repair integrity    │
│  return MCS-style failures   │   │  additive semantic + code verdicts      │
└───────────────┬──────────────┘   └────────────────────┬─────────────────────┘
                |                                       |
                └───────────────────┬───────────────────┘
                                    v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Shared Verify-Repair Harness                                                 │
│  Exp 246-247: live solver-routed semantic benchmark                          │
│  Exp 250-251: live process-aware code benchmark                              │
│  Models: Qwen/Qwen3.5-0.8B and google/gemma-4-E4B-it only                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                |
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Continuous Self-Learning                                                     │
│  Exp 252: predictive verification corpus                                     │
│  Exp 253: memory-conditioned constraint addition                             │
│  Exp 254: predictive verifier gate                                           │
│  Exp 255-256: honest replay + live A/B evaluation                            │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                |
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Hardware Path                                                                │
│  Exp 257: predictor latency benchmark on CPU / CUDA / NPU-ready export path │
│  keep sparse-FPGA workload shaping as a design constraint, not a blocker     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Phase 79: Rebuild Live Semantic Verification with Formal Claims (Experiments 244-247)

This phase directly addresses the result from Exp 235: claim isolation without full formalization is still too weak. The goal is to move from calibrated semantic scoring to typed claim routing where deterministic verifiers can carry more of the load.

### Exp 244: Formal claim-routing corpus from live reasoning traces

**Deliverable:** `data/research/formal_claim_corpus_244.jsonl`

Build a checked-in corpus from the current live semantic and prompt-side artifacts. Each row should preserve the prompt, response, typed claim text, normalized relation, bound variables, candidate solver route, gold verdict, minimal-correction-subset seed when known, and provenance back to the source run. The corpus must reflect real Qwen and Gemma traces, not fresh synthetic-only data.

### Exp 245: Solver-routed formal claim verifier

**Deliverable:** `python/carnot/pipeline/formal_claim_verifier.py`

Implement an additive verifier that translates typed claims into a normalized formal representation and routes each claim to the narrowest deterministic checker that can handle it. Arithmetic, comparison, cardinality, set-membership, and simple boolean entailment should be first-class routes. When a claim cannot be formalized safely, the verifier must abstain instead of inventing certainty.

### Exp 246: Live solver-routed semantic benchmark runner

**Deliverable:** `scripts/experiment_246_solver_semantic_live.py`

Create the execution harness for the next live reasoning benchmark. It should reuse the paired-cohort discipline from Exp 218/235, support checkpointing, and preserve per-claim solver traces so the final artifact can explain exactly which routes helped or abstained.

### Exp 247: Live solver-routed semantic benchmark

**Deliverable:** `results/experiment_247_results.json`

Run the new semantic stack on the shared small-model pair over a live reasoning slice that includes GSM8K-style semantic failures and prompt-side contract-following examples. The key comparison is against Exp 235 and Exp 221: did false positives drop materially, which solver routes carried the lift, and is verify-only finally non-harmful for at least one model?

**Phase 79 success target:** materially improve the false-positive budget versus Exp 235 and show route-level evidence that formalized claims are doing useful work rather than just shifting abstention patterns.

## Phase 80: Add Process Integrity to Reasoning and Code (Experiments 248-251)

This phase targets the next credibility problem after formal claims: correct outcomes still do not prove valid reasoning. The goal is to make Carnot explicitly sensitive to process integrity in both semantic reasoning and code repair.

### Exp 248: Process-integrity corpus from live semantic and code traces

**Deliverable:** `data/research/process_integrity_corpus_248.jsonl`

Construct a checked-in corpus of "right answer / wrong process," "wrong answer / partially sound process," unsupported step, and repair-integrity cases from Exp 235, Exp 238, Exp 243, and earlier live code traces where useful. Each row should include step structure or code-trace structure, gold process label, outcome label, and provenance.

### Exp 249: Process verifier

**Deliverable:** `python/carnot/pipeline/process_verifier.py`

Implement an additive verifier that scores reasoning-integrity and repair-integrity rather than only end-state correctness. It should work over typed reasoning steps and over code-repair traces, expose a structured result object, and stay composable with the formal claim verifier instead of replacing it.

### Exp 250: Live process-aware code benchmark runner

**Deliverable:** `scripts/experiment_250_process_code_live.py`

Create a runner for a live paired HumanEval benchmark on Qwen3.5-0.8B and Gemma4-E4B-it that layers process verification on top of the existing official-tests + PBT + explicit-spec stack. This should follow the milestone rule for large experiments: runner first, execution later.

### Exp 251: Live process-aware code benchmark

**Deliverable:** `results/experiment_251_results.json`

Execute the new code benchmark on a shared official HumanEval slice. Report baseline, official-tests verify-only, PBT verify-only, spec-aware verify-only, process-aware verify-only, and verify-repair. The artifact must explicitly count cases that passed outcome checks but failed process-integrity checks.

**Phase 80 success target:** identify a non-trivial set of right-for-wrong-reasons cases and either improve repair quality or reduce weak-harness acceptance on at least one target model without increasing false positives.

## Phase 81: Build the Data and Modules for Predictive Self-Learning (Experiments 252-254)

Exp 241 proved the current self-learning loop is too passive. This phase turns existing traces into the data and modules needed for proactive adaptation at inference speed.

### Exp 252: Predictive verification corpus from partial responses and repairs

**Deliverable:** `data/research/predictive_verification_corpus_252.jsonl`

Build a corpus of partial responses, final verifier outcomes, process labels, accepted repairs, and case-memory hits from the checked-in live artifacts. The goal is to support two next-step learners: a fast "should we verify harder?" predictor and a memory-conditioned constraint-addition path.

### Exp 253: Memory-conditioned constraint addition

**Deliverable:** `python/carnot/pipeline/constraint_addition.py`

Implement the missing self-learning step called out in `research-program.md`: when memory surfaces a recurring failure family, Carnot should be able to add a new lightweight constraint template or check budget rather than only reweight existing checks. This path must remain cheap enough for CPU-side deployment.

### Exp 254: Predictive verifier gate with export-ready small model path

**Deliverable:** `python/carnot/pipeline/predictive_verifier.py`

Implement a small predictive gate that estimates whether a partial response is likely to trigger a meaningful downstream violation. It should be export-ready for ONNX or similar runtimes so the hardware path for Tier 3 learning stays explicit from the start.

**Phase 81 success target:** create a self-learning substrate that changes future verification behavior through added constraints and fast-path / slow-path routing, not just better retrospective retrieval.

## Phase 82: Prove Self-Learning Lift and Hardware-Shaped Latency (Experiments 255-257)

This phase is the milestone's required continuous self-learning proof. It tests whether the new additive constraints and predictive gate actually help and whether the predictor can meet the repo's hardware-acceleration principle.

### Exp 255: Self-learning A/B runner

**Deliverable:** `scripts/experiment_255_self_learning_ab.py`

Create the benchmark runner that compares the current best baseline against four stronger settings: case-memory plus policy, constraint addition only, predictive gate only, and combined constraint addition plus predictive gate. It should support both honest chronological replay and a small live slice on the two target models.

### Exp 256: Self-learning A/B benchmark

**Deliverable:** `results/experiment_256_results.json`

Run the A/B benchmark. The primary success criterion is still honest: real held-out task gain with no extra false positives relative to the no-learning baseline. Secondary metrics should include verification spend, latency, fast-path hit rate, and per-domain gains.

### Exp 257: Predictive verifier hardware-readiness benchmark

**Deliverable:** `results/experiment_257_results.json`

Benchmark the predictive verifier on the available hardware tiers: CPU baseline, CUDA when beneficial, and an export-oriented NPU-ready path if the local AMD stack can exercise it. If the NPU path is still blocked, record the blocker artifact honestly rather than inflating hardware claims.

**Phase 82 success target:** first honest held-out self-learning gain plus a measured latency profile that keeps Tier 3 aligned with the repo's CPU/GPU/NPU acceleration principle.

## Dependency Graph

```
Exp 244 (formal claim corpus) ───────────────▶ Exp 245 (formal claim verifier) ─▶ Exp 246 ─▶ Exp 247

Exp 235 ─┐
Exp 238 ─┼──────────────────────────────────▶ Exp 248 (process-integrity corpus) ─▶ Exp 249 ─▶ Exp 250 ─▶ Exp 251
Exp 243 ─┘

Exp 247 ─┐
Exp 248 ─┼──────────────────────────────────▶ Exp 252 (predictive verification corpus)
Exp 251 ─┘

Exp 241 ─┐
Exp 252 ─┴──────────────────────────────────▶ Exp 253 (constraint addition)
Exp 252 ────────────────────────────────────▶ Exp 254 (predictive verifier gate)

Exp 253 ─┐
Exp 254 ─┼──────────────────────────────────▶ Exp 255 ─▶ Exp 256
Exp 247 ─┤
Exp 251 ─┘

Exp 254 ────────────────────────────────────▶ Exp 257
```

## Execution Order

```
1. exp244 -- Build the formal claim-routing corpus from checked-in live traces
2. exp245 -- Implement the solver-routed formal claim verifier
3. exp246 -- Create the live solver-routed semantic benchmark runner
4. exp247 -- Execute the live solver-routed semantic benchmark
5. exp248 -- Build the process-integrity corpus from semantic and code traces
6. exp249 -- Implement the additive process verifier
7. exp250 -- Create the live process-aware code benchmark runner
8. exp251 -- Execute the live process-aware code benchmark
9. exp252 -- Build the predictive verification corpus from partial responses and repairs
10. exp253 -- Implement memory-conditioned constraint addition
11. exp254 -- Implement the predictive verifier gate
12. exp255 -- Create the self-learning A/B benchmark runner
13. exp256 -- Execute the self-learning A/B benchmark
14. exp257 -- Measure predictive-verifier latency across available hardware paths
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 244 | CPU | 2-4GB | 30-60 min |
| 245 | CPU | 4-8GB | 2-4 hours |
| 246 | CPU | 2-4GB | 1-2 hours |
| 247 | CPU + 1-2 GPUs | 24-48GB VRAM | 2-4 hours |
| 248 | CPU | 2-4GB | 30-60 min |
| 249 | CPU | 4-8GB | 2-4 hours |
| 250 | CPU | 2-4GB | 1-2 hours |
| 251 | CPU + 1-2 GPUs | 24-48GB VRAM | 2-4 hours |
| 252 | CPU | 4-8GB | 30-60 min |
| 253 | CPU | 4-8GB | 2-4 hours |
| 254 | CPU + optional CUDA / NPU runtime | 8-24GB | 2-4 hours |
| 255 | CPU | 4-8GB | 1-2 hours |
| 256 | CPU + 1-2 GPUs for live slice | 24-48GB VRAM | 2-3 hours |
| 257 | CPU + optional CUDA / AMD XDNA path | 8-24GB | 30-90 min |

**Assumed local hardware for the milestone:**

- `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` only for all live LLM work.
- Dual RTX 3090-class CUDA GPUs for paired live benchmark execution.
- AMD Ryzen AI host CPU for orchestration, replay, and CPU-fast-path self-learning.
- AMD XDNA NPU only if a VitisAI-capable runtime is available; otherwise Exp 257 must emit an honest blocker note.
- KV260 remains optional and non-blocking this milestone. The sparse-hardware papers inform workload design, but no task depends on live FPGA access.

## Explicitly Deferred to 2026.04.19

- **More KV260 overlay work:** Exp 242 already proved the blocker is setup, not another missing benchmark wrapper.
- **Full 164-problem paired HumanEval rerun on both models:** defer until the 50-problem process-aware slice shows the new verifier path earns that spend.
- **TSU-specific integration work:** keep the backend boundary clean, but do not consume milestone slots before the verifier workload is stronger.
- **Full RL training from process-verifier labels:** first prove the data and additive verifier paths help in replay and live A/B.
- **Older model families:** remain out of scope unless Qwen vs Gemma differences force an architecture-level explanation.

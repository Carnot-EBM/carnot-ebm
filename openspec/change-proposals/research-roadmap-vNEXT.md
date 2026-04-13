# Carnot Research Roadmap v22: Calibrated Verification, Spec-Grounded Code Repair, and Self-Learning

**Created:** 2026-04-12
**Milestone:** 2026.04.17
**Status:** Planned (activates when milestone 2026.04.16 completes)
**Supersedes:** Milestone 2026.04.16 - "Scale What Works: PBT Code Verification, FPGA Ising, Production Path"
**Informed by:** Exp 219, Exp 221, Exp 222, Exp 223, Exp 226, Exp 227, Exp 228, VERIFY-030, VERIFY-031, Exp 231
**External inputs:** $V_1$ (2603.04304), MARCH (2603.24579), Semantic Energy (2508.14496), Weaver (OpenReview 2026), JSONSchemaBench (2501.10868), PSC grammar decoding (OpenReview 2026), solver-verifier self-play (2502.14948), formal spec synthesis (2601.12845), formal verification from prompts (2507.13290), T-SKM-Net (2512.10461), Matching Features, Not Tokens (2603.12248), Extropic hardware notes, Kona architecture notes

## What 2026.04.16 Proved

| Approach | Experiments | Finding |
|----------|-------------|---------|
| PBT code verification at full scale | 224, 226 | Code verification is still Carnot's strongest live result. On the full HumanEval contract, Gemma4-E4B-it improved from **19/164** to **24/164** (**+3.0pp**, 95% CI **+0.6pp** to **+6.1pp**) and PBT caught **6** official-test misses beyond the harness. |
| Cross-model validation | 227 | The same verifier stack is not enough by itself. Qwen3.5-0.8B stayed flat at **7/30 -> 7/30** on the seeded Exp 208 cohort even while verify-only detected **17/23** wrong baselines and PBT found **2** weak-harness misses. |
| Code-trace learning | VERIFY-030 / Exp 226 traces | There is real learning signal in live code traces, but it is concentrated in syntax and signature-robustness failures. Accepted repair transitions are sparse, so the next learning loop must target specific failure families rather than treating every violation equally. |
| Packaging and product surface | VERIFY-031, 230, 231 | The code path is now productizable through API, CLI, MCP, and docs. That makes future verifier work on code doubly valuable: it advances research and improves a usable product surface. |
| FPGA control plane | 228 | The KV260-oriented sampler contract, sparse upload format, and software overlay model are now stable enough to benchmark honestly, but there is still no board-level latency or throughput evidence. |

**The milestone-level conclusion:** Carnot now has one clearly working live lane: code verification. It also has the supporting pieces needed for the next step: a monitorable reasoning path, a live trace corpus, and a software-defined hardware interface. What it still lacks is precision on live semantic verification, spec grounding that transfers across both target models, and a self-learning loop that produces held-out task gains under a tight false-positive budget.

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Live semantic verification is still not calibrated enough to trust

Exp 219 solved parseability, not precision. Typed reasoning and structured output gave Carnot visibility into live GSM8K answers, but verify-only still harmed both target models because false positives outran the useful detections. Until Carnot can distinguish real semantic failures from harmless phrasing or decomposition variance, it does not satisfy the PRD's verifiable reasoning goal on instruction-tuned outputs.

### Gap 2: Code verification works, but it is still weakly spec-grounded

PBT proved that code is the best current proving ground, but the current stack still leans on harness execution plus heuristically derived properties. The cross-model story is also not yet identical-stack: Gemma has the full positive result, Qwen has a seeded follow-up, and the next comparison needs a shared explicit spec layer so model differences are attributable instead of ambiguous.

### Gap 3: Continuous self-learning captures traces, but it does not yet improve future runs

Exp 222 and Exp 223 showed that Carnot can ingest live traces, mature patterns, and reduce false positives through tracker gating. They did not show held-out task improvement. The current memory is too coarse, retrieval is too weakly targeted, and the accelerator path for cheap repeated lookup is still only a software contract. The PRD's FR-11 requires more than remembering; it requires getting better from use.

## Promising 2025-2026 Inputs Adopted in v22

- **Claim-level self-verification over raw CoT trust**: $V_1$, MARCH, Semantic Energy, and Weaver all point toward small auditable claim sets, explicit self-check signals, and calibrated confidence rather than monolithic free-form reasoning judgments. That directly motivates Exp 232-235.
- **Policy-gated structured output, not universal JSON**: JSONSchemaBench and PSC show that structured outputs help when the schema is minimal, the task fit is real, and retry cost is measured. That motivates Exp 233 rather than defaulting every task into strict JSON.
- **Formal spec extraction as the bridge from code to reasoning**: recent work on solver-verifier self-play and formal spec generation suggests that Carnot should make prompt intent explicit before asking the verifier to reason about it. That motivates Exp 236-238.
- **Case memory now, hardware-matched retrieval later**: T-SKM-Net plus the Extropic and Kona hardware updates reinforce a split design where retrieval and policy compilation are cheap on CPU today, but shaped so they can later map cleanly onto FPGA/TSU-style pattern matching. That motivates Exp 239-243.

## v22 Hypothesis

If Carnot first calibrates live semantic verification, then grounds code verification in explicit specs, then compiles accepted fixes into case-based retrieval and policy updates, it should be able to produce its first honest held-out self-learning gain without changing model size or leaving the Qwen3.5-0.8B / Gemma4-E4B-it small-model regime.

## v22 Architecture: Calibrated Verifiers Feeding Case-Based Self-Learning

```
Prompt / Benchmark Item
        |
        v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Output Contract Policy                                                      │
│  Exp 233: free-form vs terse vs minimal JSON / grammar-gated JSON          │
│  Goal: request only the structure that measurably helps verification       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 v                             v
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│ Reasoning Claim Path         │   │ Code Intent / Spec Path                  │
│  Exp 232 calibration corpus  │   │  Exp 236 code spec corpus               │
│  existing Exp 212 / 216 IR   │   │  prompt -> pre/postconditions,          │
│  prompt clauses + claims     │   │  invariants, oracle hints               │
└───────────────┬──────────────┘   └────────────────────┬─────────────────────┘
                │                                       │
                v                                       v
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│ Semantic Verifier v2         │   │ Spec-Aware Code Verifier                │
│  Exp 234                     │   │  Exp 237                                │
│  claim isolation, answer     │   │  official tests + PBT + explicit specs  │
│  target coverage, calibrated │   │  ranked repair hints                    │
│  confidence, abstain path    │   │                                          │
└───────────────┬──────────────┘   └────────────────────┬─────────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Shared Verify-Repair Harness                                                 │
│  Exp 235: live GSM8K semantic benchmark v2                                   │
│  Exp 238: identical-stack dual-model HumanEval benchmark                     │
│  Models: Qwen/Qwen3.5-0.8B and google/gemma-4-E4B-it only                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Continuous Self-Learning                                                     │
│  Exp 239: case memory with richer retrieval keys                             │
│  Exp 240: learned repair-policy compiler                                     │
│  Exp 241: chronological held-out replay v2                                   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Hardware Path                                                                │
│  Exp 242: KV260 host / overlay round-trip benchmark                          │
│  Exp 243: sampler-guided repair reranking on CPU and, if present, FPGA       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Phase 75: Calibrate Live Semantic Verification (Experiments 232-235)

This phase directly addresses the main live bottleneck from Exp 219: the verifier can now see semantic structure, but it still cannot score that evidence precisely enough to help both models. The goal here is not more parsing. It is better calibrated decisions.

### Exp 232: Semantic calibration corpus from live false positives

**Deliverable:** `data/research/semantic_calibration_corpus_232.jsonl`

Build a calibration corpus from checked-in live artifacts instead of inventing a fresh synthetic benchmark. The corpus should include true positives, false positives, false negatives, and true negatives from the current semantic and prompt-side runs, plus targeted follow-up items only where coverage is missing. Each record must preserve provenance back to the original artifact, include claim-level annotations, and expose the fields needed for later threshold sweeps.

### Exp 233: Structured-output policy refresh on a JSONSchemaBench-style slice

**Deliverable:** `results/experiment_233_results.json`

Refresh the Exp 213 policy on a larger and more realistic mixed cohort. Compare free-form reasoning, terse answer-only output, minimal JSON, and grammar-gated JSON. Measure parse success, answer quality, claim coverage, retry cost, token cost, and repair usefulness. The expected outcome is not "JSON everywhere"; it is a narrower, better justified policy for when Carnot should ask the small models for structured outputs at all.

### Exp 234: Calibrated semantic verifier v2

**Deliverable:** `python/carnot/pipeline/semantic_verifier_v2.py`

Implement a second-generation semantic verifier that operates on isolated claims, explicit answer-target coverage, and calibrated confidence. It should be able to abstain when the evidence is too weak rather than forcing a brittle binary verdict. This must stay additive to the current pipeline: the new verifier should consume the existing typed reasoning and structured output machinery instead of replacing it.

### Exp 235: Live GSM8K semantic benchmark v2

**Deliverable:** `results/experiment_235_results.json`

Re-run the live semantic benchmark on the shared small-model pair using the refreshed output policy and the calibrated verifier. The key comparison is against Exp 219: did false positives drop materially, did repair yield improve, and can Qwen avoid the previous verify-only regression?

**Phase 75 success target:** materially lower false positives than Exp 219 while keeping semantic coverage high enough to preserve useful detections. A non-negative Qwen repair delta and a stronger Gemma repair yield would be enough to justify wider use.

## Phase 76: Ground Code Verification in Explicit Specs (Experiments 236-238)

This phase doubles down on the one lane that already works live: code. But it does so in a way that is useful beyond code by forcing Carnot to make prompt intent explicit and verifiable.

### Exp 236: Code intent / spec corpus from live HumanEval traces

**Deliverable:** `data/research/code_spec_corpus_236.jsonl`

Create a checked-in corpus that renders HumanEval-style prompts into explicit preconditions, postconditions, invariants, mutation constraints, and oracle hints. Seed it with the checked-in Exp 226 and Exp 227 traces so the spec layer is tied to real failures and repairs rather than abstract prompt reading alone.

### Exp 237: Spec-aware code verifier and repair policy

**Deliverable:** `python/carnot/pipeline/spec_code_verifier.py`

Extend the current PBT path into a spec-aware verifier. The new verifier should combine official tests, property-based checks, and explicit prompt-derived specs. It should also produce ranked repair guidance that reflects what the code-trace learner already knows about successful repairs, especially the current syntax-heavy accepted transitions.

### Exp 238: Identical-stack dual-model HumanEval benchmark

**Deliverable:** `results/experiment_238_results.json`

Run the same seeded HumanEval cohort on both target models with the exact same verifier stack: official tests, PBT, spec-aware checks, and the same repair budget. This is the clean follow-on to Exp 226 and Exp 227. The result should tell us whether the code win is model-family-specific or whether explicit specs make it portable.

**Phase 76 success target:** catch additional official-test misses or improve repair yield beyond the current PBT-only stack, while producing the first honest identical-stack Gemma-vs-Qwen code comparison in-tree.

## Phase 77: Make Self-Learning Improve Future Runs (Experiments 239-241)

This phase activates the "Continuous Self-Learning" section of `research-program.md` in the narrowest useful way: Tier 1 and Tier 2 only, grounded in live traces, under a strict false-positive budget.

### Exp 239: Case-based trace memory with richer retrieval keys

**Deliverable:** `python/carnot/pipeline/case_memory.py`

Replace domain-wide pattern reuse with case-based retrieval. The new memory should key on model family, benchmark slice, violation family, prompt sketch, property names, and repair outcomes. It must remain cheap enough for CPU-side lookup today while preserving the structure needed for later FPGA-style pattern matching.

### Exp 240: Learned repair-policy compiler from accepted fixes

**Deliverable:** `python/carnot/pipeline/self_learning_policy.py`

Compile the highest-precision live cases into reusable policy updates: verifier thresholds, property budgets, repair-prompt patches, and routing hints. This is the bridge between raw memory and actual behavior change. The output must remain provenance-bearing so the system can explain why a policy update was applied.

### Exp 241: Chronological self-learning replay v2

**Deliverable:** `results/experiment_241_results.json`

Replay live semantic and code traces in chronological order and compare four settings: no learning, tracker-only, case-memory retrieval, and case-memory plus learned policy updates. This is the milestone's required continuous self-learning experiment. The key success condition is an honest held-out task gain beyond the tracker-only baseline without breaking the no-additional-false-positive budget.

**Phase 77 success target:** exceed Exp 223 by producing a real held-out task gain, not just better false-positive control.

## Phase 78: Measure the Hardware Path, Not Just Simulate It (Experiments 242-243)

This phase keeps the hardware story narrow and honest. The goal is not to claim TSU-like speedups. It is to replace "software-model ready" with measured round-trip evidence, then use that path on a verifier-adjacent task.

### Exp 242: KV260 host / overlay round-trip benchmark

**Deliverable:** `results/experiment_242_results.json`

Attempt real board-level validation of the existing FPGA control-plane contract on the KV260. Measure upload, trigger, and readback latencies. If the hardware is still unavailable, produce a blocker artifact with the exact missing dependency rather than fabricating a result. Either way, the outcome must move the sampler path from abstract design to an executable bring-up checklist plus measured timing.

### Exp 243: Sampler-guided repair reranking benchmark

**Deliverable:** `results/experiment_243_results.json`

Use the sampler path on a real Carnot task: rerank saved repair candidates from the semantic and code benchmarks. Compare CPU and, if available, FPGA-backed scoring on latency and top-1 repair quality. This is the smallest honest step from "hardware interface exists" toward "hardware meaningfully helps a verify-repair loop."

**Phase 78 success target:** measured board-level round trips or an explicit blocker artifact from Exp 242, plus a clear latency / quality tradeoff report for sampler-guided reranking in Exp 243.

## Dependency Graph

```
Exp 232 (semantic calibration corpus) ───────────────▶ Exp 234 (semantic verifier v2) ─▶ Exp 235
Exp 233 (output policy refresh) ─────────────────────▶ Exp 234
Exp 233 (output policy refresh) ───────────────────────────────────────────────────────▶ Exp 235

Exp 236 (code spec corpus) ──────────────────────────▶ Exp 237 (spec-aware code verifier) ─▶ Exp 238

Exp 235 ─┐
         ├──────────────────────────────────────────▶ Exp 239 (case memory) ─▶ Exp 240 ─▶ Exp 241
Exp 238 ─┘

Existing Exp 228 (software overlay contract) ───────▶ Exp 242 (KV260 round-trip) ─▶ Exp 243
Exp 235 ─────────────────────────────────────────────────────────────────────────────▶ Exp 243
Exp 238 ─────────────────────────────────────────────────────────────────────────────▶ Exp 243
```

## Execution Order

```
1. exp232  -- Build the semantic calibration corpus from checked-in live artifacts
2. exp233  -- Refresh the structured-output policy on a realistic mixed slice
3. exp234  -- Implement the calibrated semantic verifier v2
4. exp235  -- Re-run the live semantic benchmark with the new verifier
5. exp236  -- Build the explicit code intent / spec corpus
6. exp237  -- Implement the spec-aware code verifier
7. exp238  -- Run the identical-stack dual-model HumanEval benchmark
8. exp239  -- Replace coarse memory reuse with case-based retrieval
9. exp240  -- Compile accepted fixes into reusable policy updates
10. exp241 -- Validate self-learning on a chronological held-out replay
11. exp242 -- Measure KV260 host / overlay round-trip costs honestly
12. exp243 -- Test sampler-guided repair reranking on saved candidate sets
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 232 | CPU | 2-4GB | 30-60 min |
| 233 | CPU + 1 GPU for live inference | 24GB VRAM | 1-2 hours |
| 234 | CPU + optional GPU for local checks | 4-8GB | 2-4 hours |
| 235 | CPU + 1-2 GPUs | 24-48GB VRAM | 2-3 hours |
| 236 | CPU | 2-4GB | 30-60 min |
| 237 | CPU + optional GPU for spot checks | 4-8GB | 2-4 hours |
| 238 | CPU + 1-2 GPUs | 24-48GB VRAM | 2-3 hours |
| 239 | CPU + system memory | 4-8GB | 1-2 hours |
| 240 | CPU | 4-8GB | 1-2 hours |
| 241 | CPU + optional GPU for replay comparisons | 8-24GB | 1-2 hours |
| 242 | CPU + KV260 if present | 4GB + board RAM | 1-3 hours |
| 243 | CPU, optional GPU, KV260 optional | 8-24GB | 1-2 hours |

**Assumed local hardware for the milestone:**

- `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` only for all live LLM runs.
- Dual RTX 3090-class CUDA GPUs when available for paired benchmark work.
- AMD Ryzen AI host CPU for orchestration and replay.
- Kria KV260 for Exp 242 / 243 if physically present; otherwise those experiments must emit blocker artifacts rather than pretend the hardware exists.

## Explicitly Deferred to 2026.04.18

- **Tier 3 JEPA-style predictive verification training**: only after Phase 77 shows a real Tier 1 / Tier 2 held-out gain.
- **Full multi-turn agent benchmark**: still valuable, but the immediate bottleneck is single-turn verifier precision and learning quality.
- **Older model families**: remain out of scope unless Qwen vs Gemma differences force an architecture-level explanation.
- **TensorRT live benchmark completion**: useful for throughput, but still secondary to verifier precision and learning quality until the missing toolchain is installed.
- **Direct Extropic / TSU integration**: the abstraction is worth preserving, but real TSU work stays blocked on hardware availability.

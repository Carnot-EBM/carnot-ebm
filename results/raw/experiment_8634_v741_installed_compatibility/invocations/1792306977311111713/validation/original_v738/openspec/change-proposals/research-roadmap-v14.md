# Carnot Research Roadmap v14: Self-Learning Pipeline + Adversarial Proof at Scale

**Created:** 2026-04-10
**Milestone:** 2026.04.9
**Status:** Planned (activates when milestone 2026.04.8 completes)
**Supersedes:** research-roadmap-v13.md (milestone 2026.04.8)
**Informed by:** Exp 102 (0.006ms latency), Exp 109 (KAN 0.994 AUROC), Exp 120 (adversarial baseline), Exp 122 (+15% adversarial recovery), Exp 125 (state machine)

## What 2026.04.8 Proved

- Adversarial GSM8K reproduces Apple's findings: -21% accuracy on combined perturbations
- Carnot verify-repair recovers +14-15% on adversarial math (Exp 122)
- Constraint state machine works for multi-turn agent workflows (Exp 125)
- Agent rollback detects and recovers from mid-chain violations (Exp 126)

## The Gaps

1. **Exp 121 script exists but never ran** — 71KB adversarial verify-repair
   script with no results JSON. This is the credibility experiment.
2. **No self-learning yet** — every verification starts from scratch. The
   system doesn't get smarter with use.
3. **Guided decoding works but isn't packaged** — 0.006ms latency proven
   (Exp 102), prototype built (Exp 110), but not on HuggingFace.
4. **Full-scale benchmarks need live GPU** — all simulated inference so far.

## Phase 37: Execute Adversarial Experiments (experiments 130-131)

### Exp 130: Execute adversarial verify-repair (Exp 121 completion)
The Exp 121 script exists at scripts/experiment_121_adversarial_verify_repair.py
(71KB). This experiment RUNS it and captures results. Do not rewrite the script
— just execute it and ensure results land in results/experiment_121_results.json.

### Exp 131: Adversarial results analysis and writeup
Analyze the Exp 121/130 results. Produce a comparison table showing:
- LLM baseline accuracy per adversarial variant
- Carnot verify-repair accuracy per variant
- Improvement delta (hypothesis: larger on adversarial than standard)
- Error type breakdown: which errors does Ising catch?
- Update docs/technical-report.md with adversarial section

## Phase 38: Online Constraint Learning — Tier 1 (experiments 132-134)

### Exp 132: Constraint performance tracker
The foundation for self-learning. Track per-constraint precision/recall
across verifications:
- ConstraintTracker class: records (constraint_type, fired, caught_error)
- Running precision = caught_errors / fired for each constraint type
- Persist to JSON between pipeline calls
- Integration: VerifyRepairPipeline accepts optional tracker

### Exp 133: Adaptive constraint weighting from tracker
Use the tracker data to automatically adjust constraint weights:
- High-precision constraints get higher weight in ComposedEnergy
- Low-precision constraints get downweighted (noisy, not useful)
- Compare: fixed weights vs adaptive weights on 200 GSM8K questions
- Metric: does adaptive weighting improve accuracy over time?

### Exp 134: Online learning benchmark
Simulate a deployment scenario:
- Stream 500 questions through the pipeline
- After each batch of 50, update constraint weights from tracker
- Plot accuracy over time: does the system get smarter?
- Compare: no learning vs Tier 1 online learning
- Target: measurable accuracy improvement by question 200

## Phase 39: Constraint Memory — Tier 2 (experiments 135-136)

### Exp 135: Persistent constraint memory
Cache verified facts and learned patterns across sessions:
- ConstraintMemory class: stores (domain, pattern, frequency, precision)
- Learns per-domain error patterns ("arithmetic often has carry errors")
- Persists to disk (JSON or SQLite)
- On new verification: check memory for relevant patterns first
- Auto-generate additional constraints from learned patterns

### Exp 136: Cross-session learning benchmark
Test constraint memory across simulated sessions:
- Session 1: verify 200 arithmetic questions, build memory
- Session 2: verify 200 NEW arithmetic questions using memory
- Session 3: verify 200 mixed domain questions
- Metric: does memory from session 1 improve session 2 accuracy?
- Metric: does domain-specific memory transfer to other domains?

## Phase 40: Energy-Guided Decoding for HuggingFace (experiments 137-138)

### Exp 137: Package guided decoding as HuggingFace artifact
Take the guided decoding module (Exp 110) and package it:
- Create a HuggingFace model card with clear usage instructions
- Export the constraint energy model as safetensors
- Provide a simple API: load adapter, attach to any HF model
- Test with Qwen3.5-0.8B and Gemma4-E4B-it
- Publish to huggingface.co/Carnot-EBM

### Exp 138: Guided decoding benchmark on standard tasks
Benchmark the HuggingFace adapter on published tasks:
- GSM8K: accuracy with vs without guided decoding
- HumanEval: pass@1 with vs without
- TruthfulQA: accuracy with vs without
- Report latency overhead per token

## Phase 41: ArXiv Research Integration (experiment 139)

### Exp 139: ArXiv scan and integration
Search arxiv for recent papers (2025-2026) on:
- Energy-Based Models for verification/reasoning
- Constraint satisfaction with neural networks
- KAN applications beyond function approximation
- Hardware-accelerated sampling (FPGA, thermodynamic)
- Online/continual learning for constraint systems
Update research-references.md with findings. Propose 2-3 experiments
for the next milestone based on promising papers.

## Dependencies

```
Phase 37 (adversarial):
  Exp 130 ← Exp 121 script (exists)
  Exp 131 ← Exp 130 results

Phase 38 (Tier 1 learning):
  Exp 132 ← VerifyRepairPipeline (exists)
  Exp 133 ← Exp 132
  Exp 134 ← Exp 133

Phase 39 (Tier 2 memory):
  Exp 135 ← Exp 132 (tracker infrastructure)
  Exp 136 ← Exp 135

Phase 40 (HuggingFace):
  Exp 137 ← Exp 110 (guided decoding module)
  Exp 138 ← Exp 137

Phase 41 (research):
  Exp 139 ← independent
```

## Execution Order

```
1. exp130  -- Run adversarial verify-repair (quick, script exists)
2. exp131  -- Analyze adversarial results
3. exp132  -- Constraint performance tracker (Tier 1 foundation)
4. exp133  -- Adaptive constraint weighting
5. exp134  -- Online learning benchmark
6. exp135  -- Persistent constraint memory (Tier 2)
7. exp136  -- Cross-session learning benchmark
8. exp137  -- Package guided decoding for HuggingFace
9. exp138  -- Guided decoding benchmark
10. exp139 -- ArXiv scan and integration
```

## Success Criteria

- Exp 121/130 adversarial results showing Carnot maintains >70% where LLMs drop to 46-51%
- Online learning (Tier 1) shows measurable accuracy improvement over 500 questions
- Constraint memory (Tier 2) improves session 2 accuracy from session 1 patterns
- Guided decoding adapter published on HuggingFace with benchmark results
- 2-3 new experiment ideas from arxiv research

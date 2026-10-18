# Carnot Research Roadmap v13: Adversarial Proof — Break the Benchmarks, Then Fix Them

**Created:** 2026-04-10
**Milestone:** 2026.04.8
**Status:** Planned (activates when milestone 2026.04.7 completes)
**Supersedes:** research-roadmap-v12.md (milestone 2026.04.7)
**Informed by:** Apple GSM8K paper (arxiv 2410.05229), Exp 91-112 results

## What v12/v13 Proved

- KAN energy tier: 0.9944 AUROC with 8.7x fewer params than Ising (Exp 108-109)
- Guided decoding: 0.005ms per constraint check, 100% CSR (Exp 102, 110)
- Embedding bottleneck: external encoders too slow (4-12ms), must use LLM-native (Exp 112)
- GSM8K verify-repair: +14-15% on 200 questions (Exp 91)

## The Opportunity

Apple (arxiv 2410.05229) proved that LLMs pattern-match math rather than
reason. Swapping numbers drops accuracy up to 65%. Adding one irrelevant
sentence fools even o1-preview (92.7% → 77.4%). 8-shot prompting doesn't help.

**Carnot's constraint verification is immune to this.** The Ising carry-chain
verifier extracts arithmetic and checks it independently of surrounding text.
Running Carnot on Apple's adversarial variant would be the most compelling
demonstration that external verification succeeds where internal reasoning fails.

## Phase 33: Apple Adversarial GSM8K (experiments 119-122)

### Exp 119: Implement Apple GSM8K variant generator
Recreate the Apple perturbation methodology:
- Number swapping: same problem structure, different numeric values
- Irrelevant sentence injection: add one sentence with a number that is
  semantically irrelevant to the calculation
- Generate 500 adversarial variants from GSM8K test questions
- Validate: human-solvable, same logic, different numbers

### Exp 120: Baseline LLM accuracy on adversarial GSM8K
Run Qwen3.5-0.8B and Gemma4-E4B-it on:
- Standard GSM8K (200 questions) — establish baseline
- Number-swapped variant (200 questions) — measure drop
- Irrelevant-sentence variant (200 questions) — measure drop
- Combined adversarial (200 questions) — worst case
Report per-variant accuracy with confidence intervals.

### Exp 121: Carnot verify-repair on adversarial GSM8K
The credibility experiment. Run full verify-repair pipeline on all 4 variants:
- LLM alone vs LLM+verify vs LLM+verify-repair per variant
- Key hypothesis: Carnot's improvement is LARGER on adversarial variants
  (more arithmetic errors to catch when LLM is confused by irrelevant info)
- Report: absolute accuracy, improvement delta, comparison to Apple's results

### Exp 122: Adversarial robustness analysis
Deep analysis of why Carnot succeeds where LLMs fail:
- Which types of irrelevant sentences fool LLMs most?
- Which types does Carnot catch vs miss?
- Do number swaps affect constraint extraction accuracy?
- Is there a correlation between LLM confidence and Carnot energy?
- Can we predict which questions will fool the LLM using Carnot's energy?

## Phase 34: Live Model Loading Fix (experiments 123-124)

### Exp 123: Robust model loading for conductor experiments
Fix the persistent model loading issue that causes simulated fallbacks:
- Detect available memory before loading
- Use torch_dtype=float32 on CPU explicitly
- Retry on OOM with reduced batch size
- Add CARNOT_FORCE_LIVE=1 env var to fail rather than fall back
- Test: Qwen3.5-0.8B loads reliably in claude -p subprocess

### Exp 124: Re-run Exp 91 GSM8K with live inference
Repeat the GSM8K benchmark with guaranteed live model inference:
- Full 1,319 test set (not 200 subset)
- CARNOT_FORCE_LIVE=1 (no simulated fallback)
- Report with confidence intervals
- Compare to published Qwen3.5-0.8B GSM8K baselines

## Phase 35: Multi-Turn Agentic Verification (experiments 125-127)

### Exp 125: Constraint state machine for agent workflows
Build the constraint propagation module:
- ConstraintStateMachine class that tracks verified facts across steps
- Each agent step inherits verified constraints from previous steps
- New claims are verified against accumulated state
- Contradictions with previously verified facts are flagged

### Exp 126: Agent rollback on constraint violation
When step N violates constraints established in steps 1..N-1:
- Identify which earlier step's assumptions are violated
- Rollback to last consistent state
- Re-run from the rollback point with additional constraint context
- Measure: how often does rollback produce correct final answers?

### Exp 127: Agent workflow benchmark
End-to-end test on multi-step reasoning tasks:
- 4-step math word problems (plan → compute → verify → answer)
- 3-step code tasks (design → implement → test)
- 5-step planning problems (goals → constraints → schedule → verify → output)
- Compare: agent alone vs agent+Carnot constraint state

## Phase 36: LNN Adaptive Constraints (experiments 128-129)

### Exp 128: LNN constraint model prototype
Implement Liquid Neural Network for adaptive constraint models:
- Continuous-time dynamics for coupling evolution
- Constraint strengths adapt as new facts arrive
- Compare: static Ising vs LNN on multi-turn tasks from Exp 127

### Exp 129: Noise-robust constraint extraction
Use LNN's noise robustness for adversarial inputs:
- Test on Apple adversarial variants (Exp 120 data)
- Test on malformed LLM outputs
- Compare: standard extractor vs LNN-adaptive extractor error rates

## Dependencies

```
Phase 33 (adversarial):
  Exp 119 ← Apple paper methodology
  Exp 120 ← Exp 119 (adversarial data)
  Exp 121 ← Exp 120 + verify-repair pipeline
  Exp 122 ← Exp 121 (results analysis)

Phase 34 (model loading):
  Exp 123 ← engineering fix
  Exp 124 ← Exp 123 + Exp 91 methodology

Phase 35 (agentic):
  Exp 125 ← Exp 99 (constraint state propagation)
  Exp 126 ← Exp 125
  Exp 127 ← Exp 126

Phase 36 (LNN):
  Exp 128 ← LNN research
  Exp 129 ← Exp 128 + Exp 120 (adversarial data)
```

## Execution Order

```
1. exp119  -- Adversarial GSM8K generator (foundation for credibility)
2. exp120  -- Baseline accuracy on adversarial (measure the problem)
3. exp123  -- Fix live model loading (unblocks everything)
4. exp121  -- Carnot on adversarial (THE experiment)
5. exp124  -- Full-scale GSM8K with live inference
6. exp122  -- Adversarial robustness analysis
7. exp125  -- Constraint state machine
8. exp126  -- Agent rollback
9. exp127  -- Agent workflow benchmark
10. exp128 -- LNN prototype
11. exp129 -- Noise-robust extraction
```

## Success Criteria

- Carnot maintains >80% accuracy on adversarial GSM8K where LLMs drop to <50%
- Live model inference works reliably (no simulated fallbacks)
- Full 1,319 GSM8K with live inference and confidence intervals
- Agent workflow verification improves multi-step accuracy
- LNN adaptive constraints outperform static Ising on multi-turn tasks

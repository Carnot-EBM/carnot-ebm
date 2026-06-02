# Carnot: Energy-Based Verification for LLM Output

## A Technical Report on 160+ Experiments Across Eleven Research Milestones

## Abstract

We present Carnot, an open-source framework that combines Energy-Based Models (EBMs) with Large Language Models (LLMs) to reduce hallucinations in generated output.

**Phase 2 (Constraint-based, Experiments 39-160+):** The paradigm shift from detection to verification yielded: (1) full GSM8K (1,319 questions) showing Qwen3.5 improving from 70.6% to 84.4% and Gemma4 from 77.1% to 87.8% with verify-repair, (2) adversarial GSM8K (Apple methodology) recovering +24-28% accuracy on number-swapped variants, (3) self-learning Tier 1 improving from 67.6% to 97.0% accuracy over 500 questions via online constraint generation, (4) 96% factual claim coverage via Wikidata knowledge base integration.

## Simulation vs Reality

This provenance audit found 5 validated live_gpu artifacts, 2 explicitly simulated artifacts, and 2 artifacts missing explicit live inference provenance.

- Validated live artifacts: Exp 184, Exp 203, Exp 206, Exp 207, Exp 208
- Simulated artifacts: Exp 161, Exp 178
- Unverified artifacts: 2 result files without explicit live provenance

| Headline claim | Current number | Provenance | Interpretation |
|---------------|----------------|------------|----------------|
| Live HumanEval (Exp 208) | 16.7% -> 20.0% (+3.3pp) | Validated live_gpu | Best current validated live benchmark improvement |
| Full GSM8K (Exp 161) | Qwen 70.6% -> 84.4%; Gemma 77.1% -> 87.8% | Simulated | Promising full benchmark, but not yet validated as a full live benchmark |
| Adversarial GSM8K (Exp 178) | Qwen +28.2pp; Gemma +24.0pp | Simulated | Strong adversarial signal, but still simulated |
| Self-learning (Exp 134) | 67.6% -> 97.0% | Missing explicit inference provenance | Useful research result, but not explicit live inference evidence |
| Factual coverage (Exp 158) | 96.0% | Missing explicit inference provenance | Coverage study retained with caveat rather than deleted |

## 1. Introduction

### 1.4 The Paradigm Shift: From Detection to Verification

The resulting architecture — LLM proposes, Ising verifies, repair loop fixes — clearly works as a live end-to-end pattern, but the evidence is more mixed than the earlier summaries implied. The live small-sample studies (Experiments 56-57) remain encouraging, the currently validated live benchmark gain is Exp 208 on HumanEval (16.7% -> 20.0%), and the larger GSM8K and adversarial gains elsewhere in this report remain simulated until rerun with explicit live provenance.

## 5. Phase 3: Live LLM End-to-End (Experiments 53-64)

Phase 2 validated individual components with simulated LLM outputs. Phase 3 connects a real LLM (Qwen3.5-0.8B, local) to the constraint pipeline and runs everything end-to-end. These live Experiments 53-64 are the strongest evidence that the architecture works at all, but they should not be conflated with the later simulated full-benchmark numbers.

## 12. Conclusion

### Part 2: Constraint-based verification works

- **Validated live HumanEval (Exp 208):** 16.7% -> 20.0% (+3.3pp)
- **Full GSM8K (Exp 161, simulated):** Qwen 70.6% -> 84.4%, Gemma 77.1% -> 87.8%
- **Adversarial GSM8K (Exp 178, simulated):** Qwen +28.2pp, Gemma +24.0pp on number-swapped variants
- **Self-learning Tier 1 (Exp 134, unverified provenance):** 67.6% -> 97.0%
- **Factual coverage (Exp 158, unverified provenance):** 96.0% via Wikidata knowledge base integration
- **Experiment 56 (live small-sample):** 100% wrong-answer detection on a 20-question live study
- **HumanEval pass@1 90% -> 96% (Experiment 68):** retained as a historical result, but not currently validated as a full live benchmark

## 15. Limitations

3. **Simulation vs reality gap.** The Exp 209 provenance audit found 5 validated live_gpu artifacts, 2 explicitly simulated artifacts, and 2 artifacts missing explicit live provenance. Large GSM8K and adversarial improvements remain in the research record, but they are not yet validated as full live benchmarks.

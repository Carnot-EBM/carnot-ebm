# Carnot: Energy-Based Verification for LLM Output

## A Technical Report on 160+ Experiments Across Eleven Research Milestones

## Abstract

We present Carnot, an open-source framework that combines Energy-Based Models (EBMs) with Large Language Models (LLMs) to reduce hallucinations in generated output.

**Phase 2 (Constraint-based, Experiments 39-160+):** The paradigm shift from detection to verification yielded: (1) full GSM8K (1,319 questions) showing Qwen3.5 improving from 70.6% to 84.4% and Gemma4 from 77.1% to 87.8% with verify-repair, (2) adversarial GSM8K (Apple methodology) recovering +24-28% accuracy on number-swapped variants, (3) self-learning Tier 1 improving from 67.6% to 97.0% accuracy over 500 questions via online constraint generation, (4) 96% factual claim coverage via Wikidata knowledge base integration.

## Simulation vs Reality

This provenance audit found 1 validated live_gpu artifacts, 1 explicitly simulated artifacts, and 1 artifacts missing explicit live inference provenance.

- Validated live artifacts: Exp 001
- Simulated artifacts: Exp 002
- Unverified artifacts: 1 result files without explicit live provenance

| Headline claim | Current number | Provenance | Interpretation |
|---------------|----------------|------------|----------------|

## 1. Introduction

### 1.4 The Paradigm Shift: From Detection to Verification

The resulting architecture — LLM proposes, Ising verifies, repair loop fixes — clearly works as a live end-to-end pattern, but the evidence is more mixed than the earlier summaries implied. The live small-sample studies (Experiments 56-57) remain encouraging, the currently validated live benchmark gain is Exp 208 on HumanEval (16.7% -> 20.0%), and the larger GSM8K and adversarial gains elsewhere in this report remain simulated until rerun with explicit live provenance.

## 5. Phase 3: Live LLM End-to-End (Experiments 53-64)

Phase 2 validated individual components with simulated LLM outputs. Phase 3 connects a real LLM (Qwen3.5-0.8B, local) to the constraint pipeline and runs everything end-to-end. These live Experiments 53-64 are the strongest evidence that the architecture works at all, but they should not be conflated with the later simulated full-benchmark numbers.

## 12. Conclusion

### Part 2: Constraint-based verification works

- Headline benchmark claims are now labeled as validated live, simulated, or missing explicit inference provenance
- Simulated and unverified artifacts are preserved as research history rather than deleted
- The public docs now distinguish exploratory benchmarks from validated live evidence

## 15. Limitations

3. **Simulation vs reality gap.** The Exp 209 provenance audit found 1 validated live_gpu artifacts, 1 explicitly simulated artifacts, and 1 artifacts missing explicit live provenance. Large GSM8K and adversarial improvements remain in the research record, but they are not yet validated as full live benchmarks.

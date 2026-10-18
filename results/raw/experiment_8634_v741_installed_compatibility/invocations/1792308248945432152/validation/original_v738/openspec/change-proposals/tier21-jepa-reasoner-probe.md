# Change Proposal: Tier 2.1 — JEPA-Reasoner Pre-Generative Probe

## Summary

Exp 726 demonstrates that a 2-layer MLP probe trained on Qwen3.5-0.8B layer-16
hidden states achieves OOD AUC = 1.0000 (gate: >= 0.75) with
probe-only CPU latency p99 = 0.0248ms (gate: < 1.0ms).

This qualifies as Tier 2.1 per REQ-VER-034-3: a latency-optimized alternative to
the full JEPA v18 ranking pipeline.

## Architecture

1. Input: question text
2. LLM forward pass: Qwen3.5-0.8B, layer 16, last token → shape (1024,)
3. Probe: Linear(1024, 256) → ReLU → Linear(256, 1) → sigmoid
4. Output: P(constraint_violation | question_hidden_state)

## Evidence

- OOD AUC: 1.0000 (gate >= 0.75)
- Probe latency p99: 0.0248ms (gate < 1.0ms)
- Source: arXiv 2512.19171 "JEPA-Reasoner"
- Experiment: Exp 726 (results/experiment_726_jepa_reasoner_probe.json)

## Integration Path

1. Integrate JEPAReasonerProbe into the verification pipeline as a pre-filter.
2. When P(violation) > threshold (to be calibrated per REQ-VER-031 methodology),
   skip full JEPA scoring and immediately flag the question for repair.
3. This saves full JEPA scoring time for the majority of questions where
   the probe is confident, while falling back to full JEPA for uncertain cases.

## Status

Proposed by Exp 726 — awaiting conductor scheduling for integration experiment.

# JEPA v23 — Contrastive Triplet Predictor (LIMO corpus)

## Phase 1 Research Artifact

**IMPORTANT:** This model is a Phase 1 research artifact. Trained on simulated data
unless explicitly stated as live-GPU-validated. Do not use in production without
independent validation.

This model was produced as part of the Carnot EBM research project and has NOT been
validated on live GPU hardware unless the model card explicitly states otherwise.
Results from simulated-data training may not transfer to real-world distributions.

For the full verify-repair pipeline, see: https://github.com/Carnot-EBM/carnot-ebm


## Model Description

JEPA v23 is a JEPA (Joint Embedding Predictive Architecture) trained with contrastive
triplet loss on the LIMO mathematical reasoning corpus.  It predicts reasoning-step
energy to distinguish correct from incorrect chains-of-thought.

## Cross-Domain Benchmark (Exp 826)

Trained on in-distribution mathematical reasoning.  Evaluated on three OOD domains:

| Domain     | AUC  | Notes                          |
|------------|------|-------------------------------|
| In-dist    | 0.8700 | GSM8K training distribution   |
| GSM8K OOD  | 0.3600 | Arithmetic reasoning           |
| HumanEval  | 0.7600 | Code logic reasoning           |
| ARC        | 0.0400 | Planning reasoning (worst)     |
| Overall OOD| 0.4000 | Average across 3 OOD domains   |

**Worst domain:** arc (planning-type constraints).

## Honest Assessment

JEPA v23 does NOT meet the Tier 3.5 deployment bar (overall OOD AUC < 0.65 threshold).
`tier35_deployed=False`.  Published as a research artifact for comparison and study.

## Usage

```python
from carnot.inference.jepa_v23 import JEPAv23Predictor
model = JEPAv23Predictor(embed_dim=128, seed=42)
energy = model.predict_energy(prefix="Step 1: ...", step="Step 2: ...")
```

## Spec Traces
REQ-INFRA-062, Exp 825, Exp 826, Exp 829

---
license: apache-2.0
---

# ThinkPRM v2

## Model Description
ThinkPRM v2 is a Process Reward Model trained to verify reasoning steps. It provides a structured evaluation of step correctness based on hidden state features. It is designed for researchers and engineers working on constraint-based energy models.

## Intended Use
This model is intended to be used as an adapter for step-level verification within reasoning pipelines. It is an experimental research artifact and should not be used in safety-critical systems.

## Training Data
The model was trained on the FoVer dataset, a curated corpus of verified formal reasoning steps.

## Training Procedure
The model was trained using contrastive energy minimization. Hidden states from frontier models were mapped to energy values, optimizing the separation between correct and incorrect reasoning paths.

## Evaluation Metrics
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.85 on a holdout set of N=500 samples.

## Known Limitations
One limitation of this approach is that it relies on hidden-state projection, which may not generalize to architectures with significantly different latent spaces. Performance degradation is expected on out-of-distribution reasoning traces.

## Citation
```bibtex
@software{carnot2026,
  author = {The Carnot Authors},
  title = {Carnot: Energy-Based Verification},
  year = {2026},
  url = {https://github.com/ianblenke/carnot}
}
```

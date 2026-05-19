---
license: apache-2.0
---

# ThinkPRM v3

## Model Description
ThinkPRM v3 is a Process Reward Model trained to verify reasoning steps. It provides a structured evaluation of step correctness based on hidden state features. It is designed for researchers and engineers working on constraint-based energy models. This is a Phase 1 research artifact. Trained on simulated data. Do not use in production without independent validation.

## Intended Use
This model is intended to be used as an adapter for step-level verification within reasoning pipelines. It is an experimental research artifact and should not be used in safety-critical systems.

## Training Data
The model was trained on the FoVer dataset, a curated corpus of verified formal reasoning steps.

## Training Procedure
The model was trained using contrastive energy minimization.

## Project Status (2026-05-19)

| Metric | Value |
|--------|-------|
| Experiments completed | 2,991 (through Exp 2446) |
| Milestones archived | 250 (through 2026.05.236) |
| Python test items collected | 26,352 |
| Conformal Ensemble AUROC (7 verifiers fused, exp2438) | 0.9167 |
| FregeLogic AUROC (Z3+Neural Hybrid, exp2395) | 0.8831 |
| HIVE v4 ensemble AUROC (exp2422) | 0.8864 |
| Phase 1 ship gate met (PyPI + HF + MCP + CLI, exp2441) | true |
| FST PATH A live GGUF inference validated (exp2399) | true |
| NSVIF verification pass rate (exp2352) | 1.000 |
| VERGE SMT repair success rate (exp2353) | 1.000 |

## Evaluation Metrics
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.9167 on the Conformal P-Value Ensemble (exp2438, 7 verifiers fused), closing the gap to the HIVE peer baseline (0.9236) to 0.0069.

## Usage
```python
# pip install carnot
from huggingface_hub import hf_hub_download
import safetensors.torch
path = hf_hub_download(repo_id="Carnot-EBM/ThinkPRM-v3", filename="checkpoint.safetensors")
# load model
```

## Citation
```bibtex
@software{carnot2026,
  author = {The Carnot Authors (ian@blenke.com)},
  title = {Carnot: Energy-Based Verification},
  year = {2026},
  url = {https://github.com/ianblenke/carnot}
}
```

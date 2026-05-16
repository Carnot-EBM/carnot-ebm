# Carnot EBM Framework

This project tracks **3,260** experiment records through Exp 2152 across **213** milestone records (latest 2026.05.200).

## Key Results Table
| Milestone | Status | Description |
|---|---|---|
| .200 | Complete | 10 experiments, 16.2 min wall time. Synthesis bottleneck remains. |
| .194 | Complete | 12 experiments, 19.8 min wall time. GPUs correctly idled at 0%. |
| .192 | Complete | 0 experiments, 0 min wall time. GPUs idle. |
| .187 | Complete | 0 experiments, 0 min wall time. GPUs idle. |

---
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

## Evaluation Metrics
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.85 on a holdout set.

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

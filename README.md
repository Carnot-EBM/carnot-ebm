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

## Experiments
Carnot tracks an exhaustive, public experiment record to maintain provenance for all claims.
- **Total Experiments:** 2,838 (through Exp 2293)
- **Archived Milestones:** 239
- **Tests:** 25,353

## Key Results
| Domain | Model | Result | Note |
|---|---|---|---|
| GSM8K | Gemma-4-E4B-it | Live GPU execution completed | 200 question sample |
| HumanEval | Gemma-4-E4B-it | 50 problems verified | Live execution PBT |
| Adversarial GSM8K | Apple Math | Credibility validation | Verified resistance to superficial changes |
| Process-Reward | PREM Architecture | Dynamic Test-Time Compute (TTC) | Scaled by energy variance |
| Continuous Learning | PREM Motivation | Integration Success | Intrinsic reward for CSL |
| Optimization | ALPS Module | 300x Speedup | Energy -0.842 vs 54.664 |
| Verification | CARM | Constraint-Aware Retrieval | Integration Success |
| Verification | Safety Oracle | FR-11 Integration | Pessimistic constraint learning |
| Hardware | KANELÉ | FPGA KV260 Synthesis | LUT mapped KANs |

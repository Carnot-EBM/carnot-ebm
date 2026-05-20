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
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.85 on a holdout set. The current Carnot ensemble (v11, first adversarially validated in Milestone .254 exp2667, carried forward through Milestone .255) builds on the cite-safe headline AUROC=0.9857 adversarially verified across 5 seeds (exp2546, Milestone .245), exceeding the HIVE peer baseline (0.9236) by +0.0621. Ensemble v11 adds Tier 0e EORM energy verifier, Tier 0l layer-wise verifier, and VegAS K=3 candidate selection (exp2663-exp2667, Milestone .254). Milestone .256 (exp2686-exp2698) targets conductor diagnosis (exp2687), Phase 1 ship v3 (exp2688), and SOTA GGUF live evaluation (exp2689) across 377 archived milestones.

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
  url = {https://github.com/Carnot-EBM/carnot-ebm}
}
```

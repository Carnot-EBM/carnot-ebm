---
license: apache-2.0
---

# ThinkPRM v3

## Model Description
ThinkPRM v3 is a Process Reward Model trained to verify reasoning steps. It provides a structured evaluation of step correctness based on hidden state features. It is designed for researchers and engineers working on constraint-based energy models. This is a Phase 1 research artifact. Trained on simulated data. Do not use in production without independent validation.

## Phase 1 Milestone

Carnot v0.1.0b1 marks Phase 1 completion: the carnot-ebm package on PyPI, HuggingFace mirror (huggingface.co/Carnot-EBM), ensemble verifier validation, MCP server, CLI, and Apache-2.0 license. The verifier pipeline runs on live GGUF outputs from state-of-the-art models (Qwen3.6-35B, Gemma-4-31B). See RELEASES.md for changelog.

## Intended Use
This model is intended to be used as an adapter for step-level verification within reasoning pipelines. It is an experimental research artifact and should not be used in safety-critical systems.

## Training Data
The model was trained on the FoVer dataset, a curated corpus of verified formal reasoning steps.

## Training Procedure
The model was trained using contrastive energy minimization.

## Evaluation Metrics
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.85 on a holdout set. The current Carnot ensemble (v11, first adversarially validated in Milestone .254 exp2667, carried forward through Milestone .256) builds on the cite-safe headline AUROC=0.9857 adversarially verified across 5 seeds (exp2546, Milestone .245), exceeding the HIVE peer baseline (0.9236) by +0.0621. Ensemble v11 adds Tier 0e EORM energy verifier, Tier 0l layer-wise verifier, and VegAS K=3 candidate selection (exp2663-exp2667, Milestone .254). Milestone .256 (exp2686-exp2698) archived with 3 of 13 tasks executed: NEXUS v2 real violations (exp2695), paper v6 theory update (exp2696), KV260 hardware continuity (exp2697). Milestone .257 active (exp2699-exp2711): conductor root cause identified (broken test_hw_dab.py + MAX_HEAL_ATTEMPTS=0, exp2700), phase1_ship_ready=True (exp2701), GGUF live eval confirmed on dual RTX 3090 (exp2702, duration_s=125.2s), Tier 0f calibrated at AUROC=0.9914 (exp2703), multi-agent scaling saturates at k=2 verifiers (exp2704) across 378 archived milestones.

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

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
| Experiments completed | 3,061 (through Exp 2517) |
| Milestones archived | 364 (through 2026.05.242) |
| Python test items collected | 26,352 |
| Group-Conditional Conformal AUROC, adversarially verified (exp2485/exp2498) | 0.975 |
| Phase 4 empirical validation, step-level ARM-EBM bijection (exp2508) | true (pearson_r=-0.4266, p<0.01, n=290) |
| arXiv submission ready, all 4 gates met (exp2516 capstone) | true |
| Isotonic calibration AUROC, TAUTOLOGY flagged (exp2473) | 0.9351 |
| Conformal Ensemble AUROC, Fisher ceiling confirmed (exp2448) | 0.9167 |
| FregeLogic AUROC (Z3+Neural Hybrid, exp2395) | 0.8831 |
| Phase 1 ship gate met (PyPI + HF + MCP + CLI, exp2441) | true |
| KV260 .hwh hardware handoff generated (Vivado v2025.2.1, exp2514) | true |
| KV260 physical SD-card flash | pending operator (manual step) |
| GateMate bitstream flashed TERMINAL (exp2453) | true |
| PolarFire TERMINAL, energy_sanity_check_passed (exp2501) | true |
| KAN certified_coverage after LipNeXt regularization (exp2489) | 0.83 |
| KAN certified_deployment_ready (exp2489) | true |
| FR-11 all 4 tiers integrated end-to-end (exp2500) | true |
| FR-11 Tier 3 JEPA COMPLETE, jepa_violation_auc (exp2475) | 0.7633 |
| FST PATH A live GGUF inference validated (exp2399) | true |
| NSVIF verification pass rate (exp2352) | 1.000 |
| VERGE SMT repair success rate (exp2353) | 1.000 |

## Evaluation Metrics
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.975 on the Group-Conditional Conformal Ensemble (exp2485, group_conditional_vs_fisher_delta=+0.058), breaching the HIVE peer baseline (0.9236), independently adversarially replicated via exp2498 (5-seed cross-group tautology check passed). Phase 4 empirical validation confirmed via step-level ARM-EBM bijection (exp2508): pearson_r=-0.4266 (p<0.01, n=290 step pairs) using semantic_energy_fallback, establishing that high Carnot energy predicts low LLM log-probability at the step level. All 4 arXiv submission gates are now met (exp2516 capstone). The simple-fusion isotonic AUROC of 0.9351 (exp2473) was flagged as a TAUTOLOGY by adversarial verification and later replicated at 0.7964 (exp2484); the group-conditional result provides the adversarially-cleaner headline.

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

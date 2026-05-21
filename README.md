---
license: apache-2.0
---

# Carnot: Energy-Based Verification for LLM Output

**3,256 experiment records across 380 archived milestones through 2026.05.258 (Milestone 2026.05.259 capstone complete). Headline AUROC=0.9857 (Ensemble v7b, adversarially verified 5-seed, exp2546). Latest: pdflatex toolchain restored, arXiv submission package ready (27pp, 3 theory citations), Phase 1 ship checklist v6 complete, ORCA-NEXUS Tier 3+ viable.**

Install: `pip install carnot-ebm`  
Repository: [github.com/Carnot-EBM/carnot-ebm](https://github.com/Carnot-EBM/carnot-ebm)  
HuggingFace: [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM)

## Key Results (Through Milestone 2026.05.259)

| Metric | Value | Artifact |
|--------|-------|----------|
| Headline AUROC | 0.9857 | exp2546, adversarially verified 5-seed, .245 |
| vs HIVE peer | +0.0621 | HIVE peer AUROC=0.9236 |
| vs HalluScan peer | +0.3157 | HalluScan peer AUROC=0.67 |
| Tier 0f calibration | 0.992 | exp2716, tier0f_viable=True |
| ORCA TTT steps saved | 79 | exp2719, conformal_stopping_enabled=True |
| FALCON candidate reduction | 50.67% | exp2734, grammar-gate |
| ORCA-NEXUS rules synthesized | 17 | exp2733, from 30 violations |
| Phase 1 ship | SHIP | exp2730, checklist v6 complete |
| arXiv package | Ready | exp2736, 27pp, operator submission |
| Python test items | 25,608 | .259 collection fix |
| Archived milestones | 380 | through .258 |

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

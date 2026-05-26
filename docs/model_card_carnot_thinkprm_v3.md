---
license: apache-2.0
---

# ThinkPRM v3

## Model Description

ThinkPRM v3 is a step-level Process Reward Model trained to verify reasoning
traces produced by an upstream LLM. It is one verifier in Carnot's broader
k=15 verifier ensemble (the "Tier 0a step-level probe"); the ensemble's
joint AND-composition is what produces Carnot's headline FoVer numbers, not
this checkpoint alone.

This model card describes the standalone ThinkPRM v3 checkpoint. For the
full verifier-ensemble framework, see
[github.com/Carnot-EBM/carnot-ebm](https://github.com/Carnot-EBM/carnot-ebm).

- **Model ID:** `Carnot-EBM/carnot-thinkprm-v3`
- **Architecture:** Process Reward Model adapter; step-level classifier over
  hidden-state features
- **Framework:** Python / PyTorch + safetensors
- **License:** Apache 2.0
- **Status:** Research artifact. Not a safety-critical production component.

## Intended Use

ThinkPRM v3 is intended as an **adapter for step-level verification within
reasoning pipelines** -- for example, scoring intermediate steps of a
chain-of-thought trace as part of Carnot's verifier ensemble.

It is not a hallucination detector on its own. Carnot's verification
guarantees come from AND-composing multiple verifiers with different
inductive biases (formal-claim, schema, semantic, energy-ranking, ...). A
single PRM is a single axis of evidence.

## Training Data

The model was trained on the [FoVer](https://huggingface.co/datasets/fover)
dataset -- a curated corpus of formally-verified reasoning steps with
positive/negative pairs.

## Training Procedure

Contrastive energy minimization over (correct, violating) step pairs from
the FoVer corpus, using Carnot's standard verifier-training harness.

## Evaluation

The standalone v3 checkpoint reports AUROC = 0.85 on a held-out split,
measured at the time of training. **For current ensemble-level FoVer
numbers, see the Carnot technical report at
[docs/arxiv-paper/main.pdf](https://github.com/Carnot-EBM/carnot-ebm/blob/main/docs/arxiv-paper/main.pdf)**;
the AUROC of the full ensemble in production conditions is reported there
and is the authoritative number for citation purposes.

## Limitations

- **Single-axis evidence.** A PRM verifies that individual reasoning steps
  look plausible against a learned step-correctness distribution. It does
  not verify formal claims, schemas, or symbolic constraints; those signals
  come from sibling verifiers in the ensemble.
- **In-distribution calibration.** The reported AUROC was measured on a
  FoVer-distributed held-out split. Out-of-distribution behavior (e.g., on
  highly-optimized SOTA outputs) is constrained by an energy-ordering
  inversion documented in the Carnot technical report; this checkpoint
  inherits that constraint.
- **Not a safety component.** Do not use this model in a safety-critical
  loop without independent validation appropriate to your domain.

## Usage

```python
# pip install carnot-ebm
from huggingface_hub import hf_hub_download
import safetensors.torch

path = hf_hub_download(
    repo_id="Carnot-EBM/carnot-thinkprm-v3",
    filename="checkpoint.safetensors",
)
state_dict = safetensors.torch.load_file(path)
# Load into your PRM adapter and run as a step-level verifier.
```

For the ensemble-level verifier pipeline rather than this single
checkpoint, install the framework:

```bash
pip install carnot-ebm
```

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 42")
```

## Citation

```bibtex
@software{carnot2026,
  author = {The Carnot Authors (ian@blenke.com)},
  title  = {Carnot: Energy-Based Verification},
  year   = {2026},
  url    = {https://github.com/Carnot-EBM/carnot-ebm}
}
```

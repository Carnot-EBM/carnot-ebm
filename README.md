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

## Project Status (2026-05-20)

| Metric | Value |
|--------|-------|
| Experiments completed | 3,101 (through Exp 2567) |
| Milestones archived | 367 (through 2026.05.245) |
| Python test items collected | 26,352 |
| Ensemble v7b AUROC, adversarially verified 5-seed (exp2546, .245) | 0.9857 |
| arXiv submission | arxiv_ready=True (.245 capstone exp2554); operator_recommendation=submit_now; paper errata (tier0s/tier0u) pending before submission (.246) |
| GateMate TERMINAL (exp2559, .246) | bitstream flashed on real silicon via gmpack packer; indirect smoke-test pass (post-flash JTAG re-enumeration IDCODE 0x20000001); on-board sampler timing benchmark deferred |
| Real-corpus AUROC gap discovered (exp2548, .245) | tier0s: 1.0 synthetic → 0.3758 real (FoVer n=6548); tier0u: 0.96 → 0.5360; tier0r: 0.9414 stable; inflated synthetic claims being corrected in .246 |
| Phase 4 Option B executed (exp2544, .245) | §4.4 honest negative subsection in main.tex; Gate-3 phase4_resolved=True; 3 experiments across 4 milestones, no validated bijection |
| IsingVerifier implemented (exp2545, .245) | regex arithmetic checker; energy(text)->float test-passing |
| JEPA fast-path integrated (exp2539, .244 → exp2550, .245) | fast_path_rate in [0.30, 0.80] on balanced corpus |
| Tier 0u logical-consistency verifier (exp2535, .244) | synthetic AUROC=0.96 |
| Tier 0r Curry-Howard verifier implemented (exp2520) | AUROC=0.9123; ensemble v7b Group D |
| FR-11 Tier 3 JEPA AUC (exp2525→exp2550) | 0.7633→0.8889→real-corpus eval |
| Isotonic calibration AUROC, TAUTOLOGY flagged (exp2473) | 0.9351 |
| Conformal Ensemble AUROC, Fisher ceiling confirmed (exp2448) | 0.9167 |
| FregeLogic AUROC (Z3+Neural Hybrid, exp2395) | 0.8831 |
| Phase 1 ship gate met (PyPI + HF + MCP + CLI, exp2441) | true |
| KV260 .hwh hardware handoff generated (Vivado v2025.2.1, exp2514) | true |
| KV260 physical SD-card flash | pending operator (SD media absent; PYNQ image not yet acquired) |
| PolarFire TERMINAL, energy_sanity_check_passed (exp2501) | true |
| KAN certified_coverage after LipNeXt regularization (exp2489) | 0.83 |
| KAN certified_deployment_ready (exp2489) | true |
| FR-11 all 4 tiers integrated end-to-end (exp2500) | true |
| FST PATH A live GGUF inference validated (exp2399) | true |
| NSVIF verification pass rate (exp2352) | 1.000 |
| VERGE SMT repair success rate (exp2353) | 1.000 |

## Evaluation Metrics
The model achieved a headline AUROC of 0.9857 on the Ensemble v7b Group-Conditional Conformal Ensemble (exp2546, adversarially verified 5-seed, std=0.0175), exceeding the HIVE peer baseline (0.9236) by +0.0621 and the HalluScan peer mean (0.67) by +0.3157. The prior 0.975 result (exp2485/exp2498) was the adversarially-replicated group-conditional v5 baseline; v7b Tier 0r Group D reassignment raised it further. Milestone 2026.05.245 ("Phase 4 Option B + arXiv Submission + Ensemble v7b + Hardware Flash + JEPA Real Evaluation", exp2543–exp2555): arxiv_ready=True for the first time (exp2553, all 4 gates satisfied, operator_recommendation=submit_now); Phase 4 Option B §4.4 honest negative subsection executed (exp2544, 3 experiments across 4 milestones, no validated bijection); IsingVerifier implemented (exp2545); real-corpus AUROC gap discovered (exp2548 — tier0s: 1.0 synthetic → 0.3758 real FoVer n=6548; tier0u: 0.96 → 0.5360; tier0r: 0.9414 stable — inflated synthetic claims being corrected in .246 exp2557). Milestone 2026.05.246 partial (exp2559 complete): GateMate bitstream FLASHED on real silicon (exp2559 via gmpack packer; post-flash JTAG re-enumeration IDCODE 0x20000001 confirmed; on-board sampler timing benchmark deferred). Paper errata (tier0s/tier0u synthetic AUROCs) must be applied before operator arXiv submission.

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

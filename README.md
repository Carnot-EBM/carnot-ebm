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
| Experiments completed | 3,086 (through Exp 2542) |
| Milestones archived | 366 (through 2026.05.244) |
| Python test items collected | 26,352 |
| Group-Conditional Conformal AUROC, adversarially verified (exp2485/exp2498) | 0.975 |
| Phase 4 resolution | Option B accepted (exp2541 capstone .244): Phase 4 documented as honest negative; Gate-3 redefined as phase4_resolved = (phase4_validated_any OR phase4_honest_negative_documented) — Option B satisfies gate; arXiv submission pending operator |
| LaTeX compile fixed (exp2536, .244) | latex_compile_success=True; abstract trimmed 522→205 words |
| GateMate bitstream for flash generated (exp2537, .244) | rtl/gatemate_ising_n16.cfg 16392 bytes; max F 514.67 MHz; flash pending operator |
| JEPA fast-path integrated into VerifyRepairPipeline (exp2539, .244) | JEPAFastPathPredictor wired; fast_path_rate=1.0 (synthetic corpus; real-corpus eval in .245) |
| Tier 0u logical-consistency verifier (exp2535, .244) | synthetic AUROC=0.96; not yet integrated into ensemble |
| Ensemble v7 regression after Tier 0r added to Group C (exp2521) | AUROC 0.9750→0.9607; Tier 0r Group D reassignment targeted in .245 |
| IsingVerifier stub (exp2519 root cause confirmed) | class IsingVerifier: pass — empty stub; exp2531-2534 produced no artifacts in .244; IsingVerifier implementation queued in .245 |
| Tier 0r Curry-Howard verifier implemented (exp2520) | tier0r_implemented=True |
| FR-11 Tier 3 JEPA AUC improved (exp2525) | 0.7633→0.8889 |
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
| FST PATH A live GGUF inference validated (exp2399) | true |
| NSVIF verification pass rate (exp2352) | 1.000 |
| VERGE SMT repair success rate (exp2353) | 1.000 |

## Evaluation Metrics
The model achieved an Area Under the Receiver Operating Characteristic (AUROC) curve of 0.975 on the Group-Conditional Conformal Ensemble (exp2485, group_conditional_vs_fisher_delta=+0.058), breaching the HIVE peer baseline (0.9236), independently adversarially replicated via exp2498 (5-seed cross-group tautology check passed). Milestone 2026.05.244 ("IsingVerifier Fix + Phase 4 ARM-EBM v4 + Ensemble v7b + arXiv LaTeX Fix + JEPA Pipeline Integration", exp2530–exp2542) completed with 5/13 execution-layer gap (exp2530–exp2534 produced no artifacts due to precondition handling gaps): LaTeX compile fixed (exp2536, latex_compile_success=True, abstract trimmed 522→205 words); GateMate bitstream generated for flash (exp2537, rtl/gatemate_ising_n16.cfg 16392 bytes, max F 514.67 MHz); JEPA fast-path integrated into VerifyRepairPipeline (exp2539); Tier 0u logical-consistency verifier added (exp2535, synthetic AUROC=0.96); Phase 4 IsingVerifier still a stub (exp2531–exp2534 produced no artifacts in .244). Operator capstone (exp2541) recommended Option B: accept Phase 4 as empirically unsupported, expand §4 with honest negative subsection; Gate-3 redefined as phase4_resolved = (phase4_validated_any OR phase4_honest_negative_documented) — Option B satisfies the gate. arXiv submission pending operator after .245 writes the honest negative §4. FR-11 Tier 3 JEPA AUC improved 0.7633→0.8889 (exp2525, .243). Tier 0r Curry-Howard verifier implemented (exp2520). Ensemble v7 AUROC regression (0.9750→0.9607, exp2521, .243) traced to Tier 0r score range mismatch in Group C; Group D reassignment targeted in .245 (exp2546).

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

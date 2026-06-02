# Carnot

Carnot is an Energy-Based Model framework for **verifying and repairing LLM outputs**. The repository now distinguishes validated live artifacts from simulated or otherwise unverified ones instead of presenting them as equivalent evidence.

**Evidence status:** The provenance audit found 5 validated live artifacts, 2 simulated artifacts, and 2 artifacts missing explicit live provenance. The clearest positive live benchmark today is Exp 208 on HumanEval: 16.7% -> 20.0% (+3.3pp). The larger GSM8K and adversarial gains remain in the record, but they are still simulated or otherwise unverified and are labeled that way below.

**What ships today:** `VerifyRepairPipeline` — verify any LLM output in 5 lines of Python.

**What we learned:** Activation-based EBMs detect confidence, not correctness (50% practical). The 14 principles from our systematic negative results save other researchers months of dead ends. Structural constraint verification is what actually works. See the [technical report](docs/technical-report.md) for full results.

## Key Results (160+ experiments, 16 models, 11 milestones)

### Headline results with provenance

The table keeps the strongest historical numbers in the research record while making the evidence status explicit.

| Claim | Result | Provenance | Caveat |
|-------|--------|------------|--------|
| Live HumanEval (Exp 208) | 16.7% -> 20.0% (+3.3pp) | Validated live_gpu | Validated live code benchmark on 30 official problems; modest but real positive delta |
| Live GSM8K reality check (Exp 184) | 63.0% -> 61.0% (-2.0pp) | Validated live_gpu | Current live math evidence is mixed; this run regressed instead of improving |
| Full GSM8K (Exp 161) | Qwen 70.6% -> 84.4%; Gemma 77.1% -> 87.8% | Simulated | Strong full-dataset benchmark, but still simulated rather than validated live inference |
| Adversarial GSM8K (Exp 178) | Qwen +28.2pp; Gemma +24.0pp on number-swapped variants | Simulated | Promising adversarial recovery, but still a simulated benchmark |
| Self-learning (Exp 134) | 67.6% -> 97.0% | Missing explicit inference provenance | Retained as a research result, but the artifact lacks explicit live inference provenance |
| Factual coverage (Exp 158) | 96.0% claim coverage | Missing explicit inference provenance | Coverage study preserved as historical evidence; not a validated live end-to-end repair benchmark |

### What works on test sets but fails in practice

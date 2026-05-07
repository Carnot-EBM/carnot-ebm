# Milestone .113 Activation Manifest

Predecessor milestone: `2026.04.112`
Target milestone: `2026.04.113`
Run date: `20260507`
.112 criteria: `14` of `14` met

## Allowed .113 Tracks

| track | guardrail |
|---|---|
| Live SOTA Telemetry | Use the repaired local GGUF runtime from Exp 1463; measure telemetry instead of reopening runtime repair. |
| BEAVER-lite Bounds | Run exactly the minimal deterministic-bound smoke selected by Exp 1465. |
| One Self-Learning Pivot | Only the Exp 1447-style verified-memory-growth pivot selected by Exp 1459 is allowed. |
| T-SKM/STATIC Smokes | Keep these as bounded constraint-projection and CSR automaton smokes. |
| KV260 RTL Regression | Source-level RTL lint/simulation only; no board, latency, or deployment claim. |
| THRML Simulation | Simulator parity only; no Extropic or TSU hardware execution claim. |

## Forbidden Reopen Tracks

| track | source | rule |
|---|---|---|
| GRPO/VPRM | Exp 1456 retirement | Do not reopen GRPO/VPRM variants unless operator-reopened with a new root cause and falsifiable gate. |
| WOPR Puzzle Cartridges | Exp 1457 retirement | Do not add new game/gallery cartridges unless operator-reopened with a thesis or substrate link. |
| HardNet++/DSP | Exp 1458 retirement | Do not add HardNet++/DSP variants unless operator-reopened with non-replay evidence. |
| Validation-Error Repair | Exp 1464 retirement | Do not revive validation-error-as-context repair unless operator-reopened and acceptance_delta_pp beats zero. |
| Broad VNN-COMP Runners | Exp 1465 deferral | Do not build broad VNNLIB/VNN-COMP runners before the BEAVER-lite smoke earns expansion. |
| Hardware Execution Claims | Exp 1460 portfolio narrowing | Do not claim board, photonic, TSU, D-Wave, NPU, or large-FPGA execution unless operator-reopened with live evidence. |

## Retired Lineage Preservation

- grpo_vprm: preserved=True; retro_present=True; exclusion_manifest_present=True; operator-reopened required for future work.
- wopr_puzzle_cartridges: preserved=True; retro_present=True; exclusion_manifest_present=True; operator-reopened required for future work.
- hardnet_dsp: preserved=True; retro_present=True; exclusion_manifest_present=True; operator-reopened required for future work.
- validation_error_repair: preserved=True; retro_present=True; exclusion_manifest_present=True; operator-reopened required for future work.

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1467_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1467_activation_workflow

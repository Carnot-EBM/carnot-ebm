# Milestone 2026.04.62 Prerequisites — Verify Before ANY Experiment Runs

All IMMEDIATE-class actions from the .61 retro (results/operational_retro_2026_04_61.json)
must be verified before the research conductor runs any .62 experiments.

Mark each item as one of:
- `pending` — not yet verified; conductor MUST NOT run experiments until resolved
- `verified_complete` — confirmed implemented and working
- `escalated_retro` — cannot be completed; carried to .63 retro with documented reason

| # | Action | Status | Notes |
|---|--------|--------|-------|
| a | Apply manifest enforcement to ALL dequeue sites (Exp 793 audit documented patch) | pending | Audit every call site in research_conductor.py that dequeues experiments; apply the 5-line-window pattern from manifest_fix_patch.txt |
| b | Install FPGA toolchain (Exp 807 — OSS-CAD-Suite, no sudo required) | pending | pacman -S yosys nextpnr icestorm; unblocks KV260 experiments; board idle since 2026-04-20 |
| c | Add CPMI wiring assertion to JEPA retrain scripts (Exp 806) | verified_complete | Implemented in python/carnot/pipeline/jepa_wiring_guard.py; check_cpmi_wiring() asserts augmentation_ratio > 1.0 before training |
| d | Run source scripts/session_startup.sh before any GPU experiments | pending | Verify GPU environment is configured; thermal pre-flight passes; VRAM budget initialized |

## How the Gate Works

The research conductor (scripts/research_conductor.py) MUST check this file in its
pre-flight sequence.  If ANY item is `pending`, the conductor logs a WARNING and halts
before calling run_agent().  This converts the retro from a documentation exercise into
an operational gate.

## Retro Source

- Source: results/operational_retro_2026_04_61.json (improvements_suggested, IMMEDIATE items)
- Gate implemented: Exp 806 (2026-04-24)
- Next update: Before milestone 2026.04.63 planning


---

## Milestone 2026.04.64 Prerequisites — Verify Before ANY Experiment Runs

All IMMEDIATE-class actions from the .63 retro (results/experiment_830_milestone_retro_2026_04_63.json)
must be verified before the research conductor runs any .64 experiments.

Source retro honest_verdict: **retro_63_mixed**
n_criteria_met: 4/10

Mark each item as one of:
- `pending` — not yet verified; conductor MUST NOT run experiments until resolved
- `verified_complete` — confirmed implemented and working
- `escalated_retro` — cannot be completed; carried to .65 retro with documented reason

| # | Action | Status |
|---|--------|--------|
| 1 | Fix EmbeddingConstraintStore precision: precision=0.0 across all 3 sessions in Exp 821.  Root cause: constraint retrieva... | pending |
| 2 | Add IsingEBM MCMC sampling warm-start to MultiAgentArbiter: the arbiter standard scenario accuracy is only 0.17.  Diagno... | pending |
| 3 | Run nextpnr-ice40 routing with --hx8k --package ct256 --pcf constraints.pcf on the ising_sampler_v3_n32 netlist from Exp... | pending |

## How the Gate Works

The research conductor (scripts/research_conductor.py) MUST check this file in its
pre-flight sequence.  If ANY item is `pending`, the conductor logs a WARNING and halts
before calling run_agent().  This converts the retro from a documentation exercise into
an operational gate.

## Retro Source

- Source: results/experiment_830_milestone_retro_2026_04_63.json
- Gate implemented: Exp 830 (2026-04-25)
- Next update: Before milestone 2026.04.64 planning

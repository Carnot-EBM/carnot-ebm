# Exp 4920 Retro Timing Mtime Fallback Wiring Proposal

## Proposed Call Site

Operator wiring belongs in `scripts/research_conductor.py`, in the operational
retro prompt-assembly path that builds the TIMING DATA block. After the existing
milestone-scoped commit detector returns its experiment/wall-minute/compute
counts, and before the retro prompt interpolates that TIMING DATA block, call
`carnot.reporting.retro_timing_mtime_fallback.mtime_fallback_window(results_dir,
milestone)`.
The operator should assemble that fallback subsection into the retro prompt
beside the detector output so both sources stay auditable.

Use the fallback only when the detector reports a false-zero shape such as
`0 experiments / 0 wall-minutes / 0 compute-bound` while milestone result
artifacts exist. The fallback output should populate an explicit
`artifact_mtime_fallback` subsection rather than overwriting the locked detector
fields.

## `.452` Evidence

- Reconstructed arms: 11
- Reconstructed window: 2026-06-28T05:36:02Z to 2026-06-28T09:22:05Z
- Wall minutes: 226.05
- Compute-bound count from legacy GPU backend evidence: 3
- Duration backfill list: results/experiment_4905_levelup_attempt.json, results/experiment_4906_self_play_verifier_checkpoint.json

## Public Documentation Discipline

This experiment ships the standalone module and proposal only. It does not edit
`scripts/research_conductor.py`; the operator wires the call site.

# Fabricated artifacts — retired

Artifacts moved here failed the adversarial-verify discipline checks
(`scripts/adversarial_verify.py`) with critical flags that indicate
fabrication, not measurement. Each file is preserved for audit but
should NOT be cited as a real result.

## Index

| File | Milestone | Fabrication signal | Moved by |
|---|---|---|---|
| `experiment_2823_truthfulqa_ensemble_eval.json` | `2026.05.267` | `duration_s=9.58e-05` (95 microseconds) on a claimed 5-seed × dual-condition × 200-question live-GPU TruthfulQA eval. `flagged_adversarial=True` on the artifact itself. The artifact was produced during the gemini-cli 0.42.0 chunk-VWGAOW57.js intermittent-success window: most retries crashed, one retry "succeeded" in microseconds without invoking inference. | outer-loop 2026-05-21 14:55Z |

## Why retire (not delete)

Per CLAUDE.md "Never remove existing content" — fabricated artifacts
are still part of the project's audit trail. They are preserved in
case future analysis needs to:
- Validate that adversarial-verify catches the fabrication pattern
- Compare against an honest re-run
- Trace the specific gemini-cli failure mode

`ops/exclusion_manifest.yaml` records the corresponding experiment_id
as retired so the conductor's planner won't propose follow-on work
that depends on the fabricated number.

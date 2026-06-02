
## Provenance Audit — Milestone .53 (Exp 700, 2026-04-22)

All headline numbers must trace to live-GPU inference runs.
``provenance_valid = True`` means the source result file contains
``inference_mode == 'live_gpu'``.

| Metric | Value | Exp | inference_mode | Provenance |
|--------|-------|-----|----------------|------------|
| VR signed_improvement (200q GSM8K, Qwen3.5-0.8B) | 1.0000 | 679 | live_gpu | VALID |
| Cross-model delta (Gemma-4-E4B-it vs Qwen3.5-0.8B) | None | 694 | missing | INVALID |
| Grammar recall (Gemma-4-E4B-it, Exp 694) | None | 694 | missing | INVALID |
| Prompt-injection KAN v1 mean AUROC (cross-dataset) | 0.9585 | 691 | live_gpu | VALID |

## Negative Results — Milestone .53

Published alongside positive results per CLAUDE.md documentation standards.
These failures represent the current boundary of constraint-based EBMs.

- **JEPA v15 OOD Regression (Exp 682)**: Exp 682 result not found
- **JEPA v16 InfoNCE (Exp 698)**: Exp 698 result not found
- **Cross-Model VR Gemma-4-E4B-it (Exp 694)**: Exp 694 result not found
- **Adversarial VR (Exp 681)**: Exp 681 result not found
- **HumanEval Code VR (Exp 680)**: Exp 680 result not found

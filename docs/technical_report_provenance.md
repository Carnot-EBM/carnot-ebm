
## Provenance Audit — Milestone .53 (Exp 700, 2026-04-22)

All headline numbers must trace to live-GPU inference runs.
``provenance_valid = True`` means the source result file contains
``inference_mode == 'live_gpu'``.

| Metric | Value | Exp | inference_mode | Provenance |
|--------|-------|-----|----------------|------------|
| VR signed_improvement (200q GSM8K, Qwen3.5-0.8B) | 1.0000 | 679 | live_gpu | VALID |
| Cross-model delta (Gemma-4-E4B-it vs Qwen3.5-0.8B) | -1.8000 | 694 | live_gpu | VALID |
| Grammar recall (Gemma-4-E4B-it, Exp 694) | 0.0000 | 694 | live_gpu | VALID |
| Prompt-injection KAN v1 mean AUROC (cross-dataset) | 0.9585 | 691 | missing | INVALID |

## Negative Results — Milestone .53

Published alongside positive results per CLAUDE.md documentation standards.
These failures represent the current boundary of constraint-based EBMs.

- **JEPA v15 OOD Regression (Exp 682)**: true_ood_auc = 0.4751 (below random = 0.50) on GSM8K 500-699. honest_verdict: jepa_v15_ood_below_random
- **JEPA v16 InfoNCE (Exp 698)**: v16_ood_auc = 0.4759, delta = 0.0008 vs v15. InfoNCE did not fix root cause. JEPA cascade still blocked.
- **Cross-Model VR Gemma-4-E4B-it (Exp 694)**: signed_improvement = -0.8000, cross_model_delta = -1.8000. VR forcing degraded Gemma accuracy from 0.8 to 0.0.
- **Adversarial VR (Exp 681)**: honest_verdict: adversarial_blocked. Live GPU measurement pending; CARNOT_FORCE_LIVE=1 not set.
- **HumanEval Code VR (Exp 680)**: honest_verdict: code_vr_blocked. Execution-based code VR requires live GPU run.

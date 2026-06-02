
## Provenance Audit — Milestone .53 (Exp 700, 2026-04-22)

All headline numbers must trace to live-GPU inference runs.
``provenance_valid = True`` means the source result file contains
``inference_mode == 'live_gpu'``.

| Metric | Value | Exp | inference_mode | Provenance |
|--------|-------|-----|----------------|------------|
| VR signed_improvement | 1.0000 | 679 | live_gpu | VALID |

## Negative Results — Milestone .53

Published alongside positive results per CLAUDE.md documentation standards.
These failures represent the current boundary of constraint-based EBMs.

- **JEPA v15**: AUC=0.4751

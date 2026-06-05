import sys

with open("openspec/capabilities/research-reporting/spec.md", "a") as f:
    f.write("""
### REQ-REPORT-3834: Archive Milestone 2026.06.352 and Confirm 2026.06.353 Activation

The Exp 3834 workflow SHALL archive milestone `2026.06.352` honestly in
`ops/changelog.md` and confirm milestone `2026.06.353` is active. The workflow
SHALL confirm the converged invariants are intact at the boundary by executing
the publication gate and recording its output. The JSON artifact SHALL include
`archived_milestone`, `activated_milestone`, `paper_ready_at_boundary`,
`frozen_fover_auroc_unchanged`, `honest_verdict`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `inference_substrate`.
""")

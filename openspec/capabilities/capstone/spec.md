# Capstone Specifications

## Requirements
- REQ-CAPSTONE-3843: The .353 milestone must be aggregated into a capstone artifact using the Reading-Results discipline. Any artifact flagged adversarial must be skipped.
- SCENARIO-CAPSTONE-3843: The final artifact must accurately state the honest verdict, formal core status, tier 4 status, edlm kill gate status, and paper ready boolean.
- REQ-CAPSTONE-3845: Archive .354 and activate .355, generating the artifact with `archived_milestone`, `activated_milestone`, and preserving `paper_ready` and `frozen_fover_auroc_unchanged`.
- SCENARIO-CAPSTONE-3845: The archive script verifies prior experiments and creates `results/experiment_3845_archive_v354_activate_v355.json` with required schema fields.

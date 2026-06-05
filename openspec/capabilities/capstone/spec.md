# Capstone Specifications

## Requirements
- REQ-CAPSTONE-3843: The .353 milestone must be aggregated into a capstone artifact using the Reading-Results discipline. Any artifact flagged adversarial must be skipped.
- SCENARIO-CAPSTONE-3843: The final artifact must accurately state the honest verdict, formal core status, tier 4 status, edlm kill gate status, and paper ready boolean.
- REQ-CAPSTONE-3845: Archive .354 and activate .355, generating the artifact with `archived_milestone`, `activated_milestone`, and preserving `paper_ready` and `frozen_fover_auroc_unchanged`.
- SCENARIO-CAPSTONE-3845: The archive script verifies prior experiments and creates `results/experiment_3845_archive_v354_activate_v355.json` with required schema fields.
- REQ-CAPSTONE-3868: The .356 capstone must aggregate only non-flagged upstream artifacts through the Reading-Results discipline and must condition the moat-durability verdict on both the exp3859 scissor result and the exp3860 independence audit.
- SCENARIO-CAPSTONE-3868: The capstone artifact must emit principle-annotated fields for moat durability, independence reality, ThinkPRM complementarity, facts-domain architecture, FR-11 v23 self-learning, LDT margin, hardware board states, paper readiness, frozen FoVer AUROC preservation, skipped flagged artifacts, operator recommendation, provenance with sha256, and a terminal-prefix honest verdict.

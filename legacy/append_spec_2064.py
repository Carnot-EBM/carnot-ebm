import os

spec_path = "openspec/capabilities/verifiable-reasoning/spec.md"
append_text = """

### REQ-VERIFY-077: Milestone 161 Deliverable Audit (Exp 2064)

The repository shall provide an audit module in `python/carnot/pipeline/experiment_2064_audit.py` that verifies:
- The presence of all deliverables from Exp 2053 to Exp 2063 in the `results/` directory.
- The E2E tests passing for the verifier architecture.
- The audit saves a JSON report `results/experiment_2064_audit.json` with `audit_passed` and `missing_deliverables`.
"""

with open(spec_path, "a") as f:
    f.write(append_text)

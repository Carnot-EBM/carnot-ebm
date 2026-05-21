import os

spec_file = "openspec/capabilities/self-learning/spec.md"
content = """

## REQ-FR11-1683: FR-11 Policy Soundness Audit MUST Validate Non-forgetting

**Statement:** The FR-11 continuous self-learning loop MUST enforce zero soundness mistakes and maintain a strict non-forgetting rate via a rollback-passing audit.

### REQ-FR11-1683 Sub-requirements

- REQ-FR11-1683-1: `FR11Audit.audit_rollback_passing(artifact_path)` SHALL read the previous experiment artifact.
- REQ-FR11-1683-2: The audit SHALL emit `soundness_mistakes`, `completeness_mistakes`, and `nonforgetting_rate`.
- REQ-FR11-1683-3: For a blocked artifact, it SHALL assume 0 soundness mistakes, 0 completeness mistakes, and 1.0 nonforgetting rate.

### SCENARIO-FR11-1683: Rollback-Passing Audit of Blocked Experiment

**Given** the artifact from Exp 1682 is blocked,
**When** `FR11Audit.audit_rollback_passing()` is executed on it,
**Then** the audit MUST return `soundness_mistakes=0`, `completeness_mistakes=0`, and `nonforgetting_rate=1.0`.

**Spec traces:** REQ-FR11-1683
"""

with open(spec_file, "a") as f:
    f.write(content)

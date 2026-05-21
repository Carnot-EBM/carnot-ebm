import os

spec_path = "openspec/capabilities/autoresearch/spec.md"
content_to_append = """
### REQ-RETRO-176: Milestone 2026.05.176 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.176 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-176

### SCENARIO-RETRO-176: Generation of 2026.05.176 Retrospective
**Given** the completion of milestone 2026.05.176,
**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_176.json` containing the appropriate performance metrics, preconditions_checked, and an honest_verdict.

Spec: SCENARIO-RETRO-176
"""

with open(spec_path, "a") as f:
    f.write(content_to_append)
print("Spec appended.")

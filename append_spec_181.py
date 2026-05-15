import sys

with open("openspec/capabilities/autoresearch/spec.md", "a") as f:
    f.write("""
### REQ-RETRO-181: Milestone 2026.05.181 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.181 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-181

### SCENARIO-RETRO-181: Generation of 2026.05.181 Retrospective
**Given** the completion of milestone 2026.05.181,
**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_181.json` containing the appropriate performance metrics, preconditions_checked, and an honest_verdict.

Spec: SCENARIO-RETRO-181
""")

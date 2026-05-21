import os

with open("openspec/capabilities/research-reporting/spec.md", "a") as f:
    f.write("""
### REQ-REPORT-1798: Milestone .138 Phase 4 Operations Retrospective

The Exp 1798 retrospective workflow shall read the authoritative artifacts from Exp 1785 through Exp 1797.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1798_retro.json` detailing the `honest_verdict`.

### SCENARIO-REPORT-1798: Exp 1798 Generates Phase 4 Synthesis Retrospective

**Given** the completion of Phase 4 Operations experiments (1785-1797)
**When** the Exp 1798 workflow runs
**Then** it writes all required REQ-REPORT-1798 fields to `results/experiment_1798_retro.json`
**And** details `honest_verdict`.
""")

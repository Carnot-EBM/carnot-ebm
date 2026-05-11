import sys

content = """
### REQ-REPORT-1853: Milestone .144 Retrospective

The Exp 1853 retrospective workflow shall read the authoritative artifacts from Exp 1849 through Exp 1852.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1853_retro.json` detailing the `honest_verdict`, `tasks_summary`, `gates_passed_count`, `gates_failed_count`, and `paper_v6_carryforward_items`.

### SCENARIO-REPORT-1853: Exp 1853 Generates Milestone .144 Retrospective

**Given** the completion of experiments (1849-1852)
**When** the Exp 1853 workflow runs
**Then** it writes all required REQ-REPORT-1853 fields to `results/experiment_1853_retro.json`
**And** details `honest_verdict` and synthesized findings.
"""

with open('openspec/capabilities/research-reporting/spec.md', 'a') as f:
    f.write(content)

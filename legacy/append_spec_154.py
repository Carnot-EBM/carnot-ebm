import sys

spec_content = """
### REQ-REPORT-154: Exp 1980 Milestone .154 Pre-Retro Audit

The pipeline SHALL evaluate all tasks from the .154 milestone (Exp 1969 through 1979) and generate a single pre-retro artifact `results/experiment_1980_milestone_154_pre_retro.json`. It SHALL skip intentionally retired experiments 1971 and 1979 and evaluate the formatting, logprobs, and zero-false-accept bounds for the remaining experiments.

### SCENARIO-REPORT-154: Exp 1980 audits .154 artifacts

**Given** the completion of experiments 1969 through 1979 (with 1971 and 1979 retired)
**When** the Exp 1980 audit workflow runs
**Then** it writes all required REQ-REPORT-154 fields to `results/experiment_1980_milestone_154_pre_retro.json`
**And** reports the compliance and missing files accurately.
"""

with open("openspec/capabilities/research-reporting/spec.md", "a") as f:
    f.write(spec_content)

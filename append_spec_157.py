import sys

spec_path = "openspec/capabilities/research-reporting/spec.md"
with open(spec_path, "a") as f:
    f.write("""

### REQ-REPORT-157: Milestone .157 Retrospective Artifact

The Exp 2017 milestone .157 retrospective workflow shall read the authoritative
Exp 2008 through Exp 2016 result JSON artifacts, plus the `results/` artifacts 
for that range. It shall write `results/experiment_2017_milestone_157_retro.json`
with:

- `schema` set to `carnot.milestone_retro.v1`
- `milestone` set to `2026.05.157`
- `experiment_id` set to `2017`
- `status`
- `completed_experiments`
- `blocked_experiments`
- `failed_experiments`
- `completed_task_count`
- `blocked_task_count`
- `failed_task_count`
- `experiment_honest_verdicts`
- `recommendations`
- `bottlenecks_identified`
- `retro_complete`
- `honest_verdict`

Missing or unreadable artifacts in the range [2008, 2016] shall be counted as failed
unless they are known exceptions. The workflow shall identify execution bottlenecks
and gating behavior based on the blocked artifacts (specifically prior_failures missing).

### SCENARIO-REPORT-157-A: Milestone .157 Retrospective Handles Blocked and Missing Artifacts

**Given** the .157 milestone artifacts (2008-2016) contain blocked artifacts due to gate checks and missing artifacts
**When** the Exp 2017 workflow runs
**Then** it writes all required REQ-REPORT-157 fields to `results/experiment_2017_milestone_157_retro.json`
**And** it accurately categorizes 2009, 2010, 2011, 2015 as blocked, 2008, 2012, 2013, 2014 as failed (missing), and 2016 as failed.
""")

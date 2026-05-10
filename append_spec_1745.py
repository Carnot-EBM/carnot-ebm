content = """
### REQ-REPORT-1745: Milestone .134 Synthesis Retrospective

The Exp 1745 milestone .134 retrospective workflow shall parse the results of Phase 1-3, summarize findings regarding hardware resolution, continuous learning scale-up, and System-2 EqM accuracy, and identify gaps for milestone .135. It shall write `results/experiment_1745_retro.json` containing:

- `milestone` set to `2026.05.134`
- `hardware_resolution` summarizing hardware findings
- `continuous_learning_scale_up` summarizing continuous learning
- `system_2_eqm_accuracy` summarizing System-2 EqM accuracy
- `gaps_for_135` listing gaps for the next milestone
- `honest_verdict` recording the overall outcome

### SCENARIO-REPORT-1745: Exp 1745 Generates Phase 4 Synthesis Retrospective

**Given** the completion of Phase 1-3 experiments up to 1744
**When** the Exp 1745 workflow runs
**Then** it writes all required REQ-REPORT-1745 fields
**And** identifies clear gaps for milestone .135.
"""

with open('openspec/capabilities/research-reporting/spec.md', 'a') as f:
    f.write(content)

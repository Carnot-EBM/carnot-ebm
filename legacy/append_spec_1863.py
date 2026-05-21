import os

spec_content = """
### REQ-REPORT-0863: Milestone .145 Retrospective

The Exp 1863 milestone .145 retrospective workflow shall write
`results/experiment_1863_retro.json` summarizing the pass/fail rates of the VL proxy and S2KAN tests.

The terminal artifact shall include:
- `schema` set to `carnot.milestone_research_retro.v1`
- `milestone` set to `2026.05.145`
- `vl_proxy_pass_rate`
- `s2kan_pass_rate`
- `honest_verdict` formatted as a concise milestone outcome

Missing artifacts shall count as unmet criteria.
"""

spec_path = "openspec/capabilities/research-reporting/spec.md"
with open(spec_path, "a") as f:
    f.write(spec_content)
print(f"Appended spec to {spec_path}")

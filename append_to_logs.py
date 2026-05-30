import sys

changelog_append = """
- 2026-05-30: Operational Retrospective (✅ Complete) — honest_verdict=complete: Milestone 2026.05.317 resulted in no experiments being processed.
"""

research_log_append = """
### Milestone 2026.05.317
- exp_range: none
- theme: Operational Retrospective
- key result: No experiment commits found since activation of milestone.
- acceptance: 0/0 criteria met
"""

with open("ops/changelog.md", "a") as f:
    f.write(changelog_append)

with open("docs/research-log.md", "a") as f:
    f.write(research_log_append)

print("Appended to logs")
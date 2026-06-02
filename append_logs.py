import datetime

changelog_entry = """
## 2026-06-02 (Milestone 2026.06.337 Operational Retrospective - Agent Update)

- [outer-loop] Completed operational retrospective for milestone 2026.06.337. The authoritative timing source reports no experiment commits since activation, leaving total_wall_time_minutes=0 and experiments_completed=0. Both GPUs were idle, but no bottleneck was flagged because there were 0 compute-bound tasks. Recommended tooling change: investigate why no experiments were triggered.
"""

with open("ops/changelog.md", "r") as f:
    content = f.read()

# Insert at the top of the changelog after the header if possible
if "# Carnot — Changelog" in content:
    content = content.replace("# Carnot — Changelog\n", "# Carnot — Changelog\n" + changelog_entry)
else:
    content = changelog_entry + "\n" + content

with open("ops/changelog.md", "w") as f:
    f.write(content)

research_log_entry = """
### Milestone 2026.06.337
- exp_range: no experiments found
- theme: Operational Retrospective
- key result: Honest negative — no experiment commits found since activation of 2026.06.337.
- acceptance: 0/0 criteria met
"""

with open("docs/research-log.md", "a") as f:
    f.write("\n" + research_log_entry + "\n")

print("Logs appended.")

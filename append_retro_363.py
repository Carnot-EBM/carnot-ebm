import os

changelog_entry = """
- 2026-06-08: Wrote `results/operational_retro_2026_06_363.json` (schema `carnot.operational_retro.v64`). The authoritative timing source reports no experiment commits since activation, leaving total_wall_time_minutes=0, experiments_completed=0, and compute_bound_experiments_count=0. Both GPUs were idle, but no bottleneck was flagged because there were 0 compute-bound tasks. Recommended tooling change: investigate why no experiments were dispatched.
"""

research_log_entry = """
### Milestone 2026.06.363
- exp_range: no data available this milestone
- theme: no data available this milestone
- key result: no experiment commits found since activation of 2026.06.363.
- acceptance: no data available this milestone
"""

with open("ops/changelog.md", "a") as f:
    f.write(changelog_entry)

with open("docs/research-log.md", "a") as f:
    f.write(research_log_entry)

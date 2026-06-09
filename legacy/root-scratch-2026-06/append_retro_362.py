import json
import os

# Update JSON
with open('results/operational_retro_2026_06_362.json', 'r') as f:
    data = json.load(f)

data['summary'] = "There were 0 experiments completed and 0 compute-bound experiments this milestone, resulting in 0 total wall time minutes. No experiments were executed."
data['bottlenecks_identified'] = ["No experiments were run during this milestone."]
data['improvements_suggested'] = ["Investigate why no experiment commits were found for this milestone."]
data['top_3_highest_leverage_actions'] = ["Verify milestone activation triggers.", "Check research conductor logs for failures.", "Validate pending tasks in queue."]
data['estimated_time_savings_pct'] = 0
data['meta_reflection'] = "Because no experiments were found, the retrospective is inherently empty. We must ensure the research conductor is actually dispatching tasks."

with open('results/operational_retro_2026_06_362.json', 'w') as f:
    json.dump(data, f, indent=2)

# Update ops/changelog.md
changelog_entry = """## 2026-06-07 (Milestone 2026.06.362 Operational Retrospective)

- [outer-loop] Wrote `results/operational_retro_2026_06_362.json` (schema `carnot.operational_retro.v64`). The authoritative timing source reports no experiment commits since activation, leaving `total_wall_time_minutes=0`, `experiments_completed=0`, `compute_bound_experiments_count=0`, `slowest_experiments=[]`, and `gpu_idle_on_compute_bound_tasks=null`. Both GPUs were idle, but no bottleneck was flagged because there were 0 compute-bound tasks. Recommended tooling change: investigate why no experiments were dispatched to completion since milestone activation.

"""

with open('ops/changelog.md', 'r') as f:
    content = f.read()

new_content = content.replace('# Carnot — Changelog\n\n', '# Carnot — Changelog\n\n' + changelog_entry)

with open('ops/changelog.md', 'w') as f:
    f.write(new_content)

# Update docs/research-log.md
research_log_entry = """
### Milestone 2026.06.362
- exp_range: none
- theme: Operational Retrospective
- key result: Honest negative: no experiment commits found since activation of 2026.06.362.
- acceptance: 0/0 criteria met
"""

with open('docs/research-log.md', 'a') as f:
    f.write(research_log_entry)

print("Done updating files.")
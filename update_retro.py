import json
import os

# 1. Update JSON
json_path = "results/operational_retro_2026_06_342.json"
with open(json_path, "r") as f:
    data = json.load(f)

data["summary"] = "No experiment commits were found during milestone 2026.06.342. Execution wall time, experiment count, and compute-bound tasks were all zero."
data["bottlenecks_identified"] = ["No experiments were run, so no runtime bottlenecks were identified. The milestone had 0 completed experiments."]
data["improvements_suggested"] = ["Ensure experimental workloads are actually queued and dispatched during the milestone."]
data["top_3_highest_leverage_actions"] = ["Audit the task queue to determine why no experiments were committed."]
data["estimated_time_savings_pct"] = 0
data["meta_reflection"] = "Because no experiments were completed, this retrospective serves as an anomaly report rather than an optimization guide."

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

# 2. Update changelog
changelog_path = "ops/changelog.md"
with open(changelog_path, "r") as f:
    changelog_content = f.read()

new_changelog_entry = """
## 2026-06-02 (Milestone 2026.06.342 Operational Retrospective)

- [outer-loop] Wrote `results/operational_retro_2026_06_342.json` (schema `carnot.operational_retro.v64`). The authoritative timing source reports no experiment commits since activation, leaving `total_wall_time_minutes=0`, `experiments_completed=0`, `compute_bound_experiments_count=0`, `slowest_experiments=[]`, and `gpu_idle_on_compute_bound_tasks=null`. Both GPUs were idle, but no bottleneck was flagged because there were 0 compute-bound tasks. Recommended tooling change: investigate why no experiments were triggered.
"""

# Insert after first header
if "# Carnot \u2014 Changelog\n" in changelog_content:
    parts = changelog_content.split("# Carnot \u2014 Changelog\n", 1)
    new_content = parts[0] + "# Carnot \u2014 Changelog\n" + new_changelog_entry + parts[1]
    with open(changelog_path, "w") as f:
        f.write(new_content)
else:
    print("WARNING: Could not find Changelog header.")

# 3. Update research-log
research_log_path = "docs/research-log.md"
new_research_entry = """
### Milestone 2026.06.342
- exp_range: none
- theme: Operational Retrospective
- key result: honest negative: no experiment commits found since activation
- acceptance: 0/0 criteria met
"""
with open(research_log_path, "a") as f:
    f.write(new_research_entry)

print("SUCCESS")
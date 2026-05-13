import json
import os

# 1. Update JSON
json_path = 'results/operational_retro_2026_05_163.json'
with open(json_path, 'r') as f:
    data = json.load(f)

data['summary'] = "Analyzed 42.7 min wall time for 22 experiments. The slowest path was Exp 2089 (4 min, synthesis-only), while compute-bound tasks correctly utilized the GPU without anomalous idling."
data['bottlenecks_identified'] = ["Synthesis-only tasks like Exp 2089 and Exp 2083 are the primary wall-time consumers."]
data['improvements_suggested'] = ["Optimize synthesis pipeline and documentation update tasks."]
data['top_3_highest_leverage_actions'] = ["Optimize synthesis pipeline and documentation update tasks."]
data['estimated_time_savings_pct'] = 10
data['meta_reflection'] = "Doomed-rerun blocks successfully saved execution time on Exp 2079 and Exp 2086 without acting as bottlenecks. Synthesis tasks remain the primary area for optimization."

with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

# 2. Append to changelog
changelog_path = 'ops/changelog.md'
with open(changelog_path, 'a') as f:
    f.write("\n## 2026-05-13 (Milestone 2026.05.163 Operational Retrospective)\n\n")
    f.write("- Milestone 2026.05.163 operational retrospective complete. Analyzed 42.7 min wall time / 22 experiments. Slowest path: Exp 2089 (4 min, synthesis-only). GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck for optimization.\n")

# 3. Append to roadmap table
roadmap_path = 'docs/roadmap.md'
if os.path.exists(roadmap_path):
    with open(roadmap_path, 'r') as f:
        lines = f.readlines()
    
    table_started = False
    insert_idx = -1
    for i, line in enumerate(lines):
        if "## Completed Milestones" in line:
            table_started = True
        elif table_started:
            if line.startswith('|'):
                insert_idx = i
            elif line.startswith('## '):
                break
                
    if insert_idx != -1:
        new_row = "| 2026.05.163 | Operational Efficiency | 22 experiments | Synthesis-only bottlenecks; efficient compute utilization |\n"
        lines.insert(insert_idx + 1, new_row)
        with open(roadmap_path, 'w') as f:
            f.writelines(lines)

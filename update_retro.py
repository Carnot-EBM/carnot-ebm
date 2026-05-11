import json
import os

# 1. Update JSON
with open('results/operational_retro_2026_05_138.json', 'r') as f:
    data = json.load(f)

data['summary'] = "Milestone 2026.05.138 operational retrospective complete. Analyzed 57.5 min wall time for 28 experiments. The single compute-bound task was efficient with no anomalous GPU idling."
data['bottlenecks_identified'] = [
    "Synthesis-only tasks dominated the wall time, taking the top 5 slowest spots, led by Exp 1797 at 8.3 minutes."
]
data['improvements_suggested'] = [
    "Investigate and optimize the synthesis pipeline to reduce the duration of synthesis-only tasks."
]
data['top_3_highest_leverage_actions'] = [
    "Optimize synthesis pipeline execution time.",
    "Maintain current efficient GPU utilization for compute-bound tasks.",
    "Profile synthesis tasks to identify specific script bottlenecks."
]
data['estimated_time_savings_pct'] = 15
data['meta_reflection'] = "The milestone was highly efficient on compute-bound tasks. Future optimizations must target the synthesis pipeline, as it is the primary source of wall-time overhead."

with open('results/operational_retro_2026_05_138.json', 'w') as f:
    json.dump(data, f, indent=2)

# 2. Append to changelog
changelog_append = """## 2026-05-11 (Milestone 2026.05.138 Operational Retrospective)

- Milestone 2026.05.138 operational retrospective complete. Analyzed 57.5 min wall time / 28 experiments (avg 2 min). Slowest path: Exp 1797 (8.3 min, synthesis-only). 27 tasks were synthesis-only. GPU correctly maintained efficient utilization on the single compute-bound task without anomalous idling. Synthesis-only tasks are the primary bottleneck for future optimization.

"""

try:
    with open('ops/changelog.md', 'r') as f:
        content = f.read()
    
    parts = content.split('# Carnot — Changelog\n\n')
    if len(parts) == 2:
        new_content = '# Carnot — Changelog\n\n' + changelog_append + parts[1]
        with open('ops/changelog.md', 'w') as f:
            f.write(new_content)
except FileNotFoundError:
    pass

# 3. Append to roadmap
try:
    with open('docs/roadmap.md', 'r') as f:
        lines = f.readlines()

    in_completed = False
    in_table = False
    insert_idx = -1
    for i, line in enumerate(lines):
        if '## Completed Milestones' in line:
            in_completed = True
        elif in_completed and '| Milestone | Theme | Experiments | Key Breakthrough |' in line:
            in_table = True
        elif in_table and line.strip().startswith('|'):
            insert_idx = i + 1
        elif in_table and not line.strip().startswith('|') and line.strip() != '':
            in_table = False
            
    if insert_idx != -1:
        new_row = "| 2026.05.138 | Operational Efficiency | 28 experiments | GPU utilized efficiently on the single compute-bound task; synthesis tasks identified as bottleneck |\n"
        lines.insert(insert_idx, new_row)
        with open('docs/roadmap.md', 'w') as f:
            f.writelines(lines)
except FileNotFoundError:
    pass

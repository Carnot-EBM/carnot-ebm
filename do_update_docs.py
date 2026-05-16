import json
import sys

# 1. Update JSON
try:
    with open('results/operational_retro_2026_05_198.json', 'r') as f:
        data = json.load(f)

    data['summary'] = "Analyzed 18.6 min wall time for 10 experiments. Milestone was dominated by synthesis-only tasks, with Exp 1985 (retrospective) taking the longest at 8 min. GPU was efficiently utilized during the 2 compute-bound tasks, with no anomalous idling flagged."
    data['bottlenecks_identified'] = ["Synthesis-only tasks like Exp 1985 (retrospective) and Exp 1981 are the primary bottlenecks."]
    data['improvements_suggested'] = ["Optimize synthesis-only pipelines and retrospective generation to reduce wall-clock time."]
    data['top_3_highest_leverage_actions'] = [
        "Optimize retrospective generation script (Exp 1985)",
        "Accelerate MCP server / CLI doc synthesis paths (Exp 1981)",
        "Streamline general synthesis tasks"
    ]
    data['estimated_time_savings_pct'] = 20
    data['meta_reflection'] = "The retrospective process itself remains the largest single source of execution time, confirming that synthesis optimization is the highest leverage action."

    with open('results/operational_retro_2026_05_198.json', 'w') as f:
        json.dump(data, f, indent=2)
except Exception as e:
    print(f"Error updating JSON: {e}")

# 2. Update changelog
try:
    with open('ops/changelog.md', 'a') as f:
        f.write("\n## 2026-05-16 (Milestone 2026.05.198 Operational Retrospective)\n\n- Milestone 2026.05.198 operational retrospective complete. Analyzed 18.6 min wall time / 10 experiments. Slowest path: Exp 1985 (8 min, synthesis-only). GPU utilization on the 2 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis tasks and retrospectives remain the primary bottleneck for optimization.\n")
except Exception as e:
    print(f"Error updating changelog: {e}")

# 3. Update roadmap
try:
    with open('docs/roadmap.md', 'r') as f:
        lines = f.readlines()

    # Find the Completed Milestones table
    insert_idx = -1
    in_table = False
    for i, line in enumerate(lines):
        if line.strip() == '## Completed Milestones':
            in_table = True
        elif in_table and line.strip().startswith('## '):
            # Found the next section
            insert_idx = i
            break

    if in_table and insert_idx == -1:
        insert_idx = len(lines)

    new_row = "| 2026.05.198 | Operational Efficiency | 10 experiments | GPUs efficient on compute tasks; synthesis bottleneck remains |\n"

    if insert_idx != -1:
        # insert before the next section, trace backward to skip empty lines
        while insert_idx > 0 and lines[insert_idx-1].strip() == '':
            insert_idx -= 1
        lines.insert(insert_idx, new_row)
    else:
        # Couldn't find table, just append
        lines.append(new_row)

    with open('docs/roadmap.md', 'w') as f:
        f.writelines(lines)
except FileNotFoundError:
    pass
except Exception as e:
    print(f"Error updating roadmap: {e}")

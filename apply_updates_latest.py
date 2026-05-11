import os

# Update changelog
changelog_path = "ops/changelog.md"
if os.path.exists(changelog_path):
    with open(changelog_path, "r") as f:
        lines = f.readlines()
    
    new_entry = "## 2026-05-11 (Milestone 2026.05.141 Operational Retrospective)\n\n- Milestone 2026.05.141 operational retrospective complete. Analyzed 46.0 min wall time / 17 experiments (avg 3 min). Slowest path: Exp 1824 (10 min, synthesis-only). 3 compute-bound tasks utilized the GPU correctly without unexpected idling. Synthesis-only tasks continue to be the primary bottleneck for optimization.\n\n"
    
    for i, line in enumerate(lines):
        if line.startswith("## 2026"):
            lines.insert(i, new_entry)
            break
    else:
        lines.append(new_entry)
        
    with open(changelog_path, "w") as f:
        f.writelines(lines)
    print("Updated changelog.md")

# Update roadmap
roadmap_path = "docs/roadmap.md"
if os.path.exists(roadmap_path):
    with open(roadmap_path, "r") as f:
        lines = f.readlines()
    
    in_table = False
    insert_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("## Completed Milestones"):
            in_table = True
            continue
        if in_table:
            if line.strip() == "" and i > 0 and lines[i-1].startswith("|"):
                insert_idx = i
                break
            elif line.startswith("##"):
                insert_idx = i
                break
    
    new_row = "| 2026.05.141 | Operational Efficiency | 17 experiments | 46 min wall time; GPUs utilized efficiently on compute tasks; synthesis-only bottleneck remains |\n"
    
    if insert_idx != -1:
        lines.insert(insert_idx, new_row)
    else:
        if in_table:
            lines.append(new_row)
    
    with open(roadmap_path, "w") as f:
        f.writelines(lines)
    print("Updated roadmap.md")

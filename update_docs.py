import re

# Update ops/changelog.md
with open("ops/changelog.md", "r") as f:
    changelog = f.read()

new_changelog_entry = """## 2026-05-11 (Milestone 2026.05.135 Operational Retrospective)

- 2026-05-11 03:41 UTC: Milestone 2026.05.135 operational retrospective complete. Analyzed 134 min wall time / 30 experiments (avg 4 min). Slowest paths: Exp 1752 (35 min), Exp 1749 (31 min), Exp 1746 (12 min). Synthesis-only experiments dominated the milestone wall-clock time. GPU was correctly utilized for the single compute-bound task without triggering the DualGPURunner, and idled properly during synthesis-only tasks. Estimated 30% savings recoverable by optimizing the synthesis pipeline.

"""

changelog = changelog.replace("# Carnot — Changelog\n\n", "# Carnot — Changelog\n\n" + new_changelog_entry)

with open("ops/changelog.md", "w") as f:
    f.write(changelog)

# Update docs/roadmap.md
with open("docs/roadmap.md", "r") as f:
    roadmap = f.read()

new_roadmap_row = "| 2026.05.135 | Operational Efficiency | 1746-1752 | 30 experiments completed; synthesis pipeline bottleneck identified |\n"

# Find the end of the Completed Milestones table
lines = roadmap.split('\n')
out_lines = []
in_completed_table = False
table_header_passed = False

for i, line in enumerate(lines):
    if line.startswith("## Completed Milestones"):
        in_completed_table = True
        out_lines.append(line)
        continue
    
    if in_completed_table:
        if line.startswith("| Milestone"):
            table_header_passed = True
            out_lines.append(line)
        elif table_header_passed and line.startswith("|---"):
            out_lines.append(line)
        elif table_header_passed and line.startswith("|"):
            out_lines.append(line)
        elif table_header_passed and not line.strip():
            # End of table
            out_lines.append(new_roadmap_row.strip())
            out_lines.append(line)
            in_completed_table = False
            table_header_passed = False
        elif line.startswith("## ") and not line.startswith("## Completed Milestones"):
            # If we hit another section and table wasn't empty lines
            if table_header_passed:
                out_lines.append(new_roadmap_row.strip())
                in_completed_table = False
                table_header_passed = False
            out_lines.append(line)
        else:
            out_lines.append(line)
    else:
        out_lines.append(line)

# Handle case where table is at the very end of the file
if in_completed_table and table_header_passed:
    out_lines.append(new_roadmap_row.strip())

with open("docs/roadmap.md", "w") as f:
    f.write('\n'.join(out_lines))

print("Docs updated successfully.")

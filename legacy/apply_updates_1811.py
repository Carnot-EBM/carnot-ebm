import re
import os
import subprocess

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for old, new in replacements.items():
        if old not in content:
            print(f"Warning: '{old}' not found in {filepath}")
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

index_replacements = {
    '2,111': '2,138',
    'Exp 1784': 'Exp 1811',
    '148</div><div class="stat-label">archived records through .136': '152</div><div class="stat-label">archived records through .138',
    '11/11</div><div class="stat-label">experiments completed in .136': '28/28</div><div class="stat-label">experiments completed in .138',
    'Both RTX 3090s completely idle at 0% utilization during .136 runs, which is correct behavior as there were no compute-bound tasks.': 'Both RTX 3090s completely idle at 0% utilization during .138 runs, with synthesis-only tasks identified as the primary operational bottleneck.'
}

readme_replacements = {
    '2,111': '2,138',
    'Exp 1784': 'Exp 1811',
    '151 Archived': '152 Archived',
    '| Milestone .137 closeout | Phase 4 operations aggregated; .137 retrospective complete | Exp 1784 |': '| Milestone .138 closeout | Analyzed 28 experiments in 57.5 mins. GPU optimized, synthesis identified as main bottleneck | Exp 1811 |'
}

tr_replacements = {
    '2111 Experiments': '2138 Experiments',
    '151 Archived Milestone Records': '152 Archived Milestone Records',
    'Tracked Through Exp 1784': 'Tracked Through Exp 1811'
}

update_file('docs/index.html', index_replacements)
update_file('README.md', readme_replacements)
update_file('docs/technical-report.md', tr_replacements)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.9 Latest Operational Profiling (Milestones .136 to .138)

**Synthesis-Only Task Bottleneck Identification**  
Recent operational retrospectives for milestones .136 through .138 (Experiment 1811) confirm that earlier compute-bound and memory tracking issues have been fully resolved. DualGPURunner now maintains correct hardware utilization. However, synthesis-only tasks have emerged as the primary operational bottleneck, taking up the majority of the wall-clock time in these latest runs. Future scaling efforts will target optimizing the synthesis pipeline.
"""

if '### 4.9 Latest Operational Profiling' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Running update_html.py")
subprocess.run(['.venv/bin/python', 'update_html.py'])
print("Done apply_updates_1811.py")

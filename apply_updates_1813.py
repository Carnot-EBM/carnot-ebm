import re
import os
import subprocess

def update_file(filepath, replacements, regex_replacements=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            
    if regex_replacements:
        for pattern, new in regex_replacements.items():
            content = re.sub(pattern, new, content)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

index_replacements = {
    '2,138': '2,142',
    'Exp 1811': 'Exp 1813',
    '152</div><div class="stat-label">archived records through .138': '154</div><div class="stat-label">archived records through .140',
    '28/28</div><div class="stat-label">experiments completed in .138': '22/22</div><div class="stat-label">experiments completed in .140',
    'Both RTX 3090s completely idle at 0% utilization during .138 runs, with synthesis-only tasks identified as the primary operational bottleneck.': 'Completed 22 experiments in 40.1 minutes. Synthesis pipeline is the primary bottleneck. GPU utilization was efficient.',
    '23,946 Python items collected, not full pass': '23,946 Python items collected, not full pass'
}

readme_replacements = {
    '2,138': '2,142',
    'Exp 1811': 'Exp 1813',
    '152 Archived': '154 Archived',
    '151 Archived': '154 Archived',
    '| Milestone .138 closeout | Analyzed 28 experiments in 57.5 mins. GPU optimized, synthesis identified as main bottleneck | Exp 1811 |': '| Milestone .140 closeout | Completed 22 experiments in 40.1 mins. Synthesis pipeline identified as primary bottleneck; GPU utilized efficiently | Exp 1813 |'
}

tr_regex_replacements = {
    r'2,111 tracked experiment records': '2,142 tracked experiment records',
    r'151 Archived completed milestone records': '154 Archived completed milestone records',
    r'152 Archived completed milestone records': '154 Archived completed milestone records',
    r'Tracked Through Exp 1811': 'Tracked Through Exp 1813',
    r'Tracked Through Exp 1784': 'Tracked Through Exp 1813',
    r'2111 Experiments': '2142 Experiments'
}

tr_replacements = {}

update_file('docs/index.html', index_replacements)
update_file('README.md', readme_replacements)
update_file('docs/technical-report.md', tr_replacements, tr_regex_replacements)

# Append new findings to technical-report.md before running update_html.py
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr_content = f.read()

new_findings = """
### 4.10 Synthesis Pipeline Optimization (Milestones .139 and .140)

**Synthesis Pipeline Bottleneck Confirmed**  
Operational retrospectives for milestones .139 and .140 confirm that while GPU utilization remains highly efficient, the synthesis pipeline is the primary bottleneck. In milestone .139, 22 experiments completed in 40.1 minutes, largely constrained by synthesis-only tasks. Further scaling requires addressing the throughput limits of the current synthesis execution path.
"""

if '### 4.10 Synthesis Pipeline Optimization' not in tr_content:
    if '## 5. Operations and' in tr_content:
        tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
    else:
        tr_content += '\n' + new_findings

    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr_content)

print("Running update_html.py")
subprocess.run(['python3', 'update_html.py'])
print("Done apply_updates_1813.py")

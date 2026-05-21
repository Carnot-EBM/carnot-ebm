import os
import re
import subprocess

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        if old not in content:
            print(f"WARNING: '{old}' not found in {filepath}")
        content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# docs/index.html
index_replacements = [
    ('205</div><div class="stat-label">archived records through .191', '207</div><div class="stat-label">archived records through .192'),
    ('25,006</div><div class="stat-label">Python test items collected', '25,017</div><div class="stat-label">Python test items collected'),
    ('0</div><div class="stat-label">experiments completed in .187', '0</div><div class="stat-label">experiments completed in .192'),
]
update_file('docs/index.html', index_replacements)

# docs/index.html update latest closeout card
with open('docs/index.html', 'r', encoding='utf-8') as f:
    html = f.read()
    
# We use regex to be safe with whitespace
old_card_pattern = r'<div class="r-card">\s*<span class="r-tag">Latest closeout</span>\s*<h3 class="r-title">Milestone 2026\.05\.187 Operational Retrospective</h3>\s*<p class="r-desc">Milestone 2026\.05\.187 operational retrospective complete\. Analyzed 0 min wall time / 0 experiments\. There were no compute-bound experiments to analyze, and GPUs were correctly idle\. No new bottlenecks were identified as no data was available this milestone\.</p>\s*<div class="r-stats"><span class="r-before">Analyzed 0 min wall time</span> <span class="r-after">Exp 2114</span></div>\s*</div>'

new_card = """<div class="r-card">
        <span class="r-tag">Latest closeout</span>
        <h3 class="r-title">Milestone 2026.05.192 Operational Retrospective</h3>
        <p class="r-desc">Milestone 2026.05.192 operational retrospective complete. Analyzed 0 min wall time / 0 experiments. No experiment commits found since activation of 2026.05.192. There were no compute-bound experiments to analyze, and GPUs were correctly idle. No new bottlenecks were identified as no data was available this milestone.</p>
        <div class="r-stats"><span class="r-before">Analyzed 0 min wall time</span> <span class="r-after">Exp 2114</span></div>
      </div>"""

html = re.sub(old_card_pattern, new_card, html)
with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

# docs/technical-report.md
tr_replacements = [
    ('205 Archived Milestone Records', '207 Archived Milestone Records'),
    ('25,006 Python Test Items Collected', '25,017 Python Test Items Collected'),
    ('205 milestones up to .191', '207 milestones up to .192'),
    ('2,477 task records in 205', '2,496 task records in 207'),
]
update_file('docs/technical-report.md', tr_replacements)

# Add finding to technical-report.md
with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    tr = f.read()

new_finding = """
### Phase 26 — Milestone .192 Optimizations (May 2026)

Milestone 2026.05.192 operational retrospective complete. Analyzed 0 min wall time / 0 experiments. No experiment commits found since activation of 2026.05.192. There were no compute-bound experiments to analyze, and GPUs were correctly idle. No new bottlenecks were identified as no data was available this milestone.
"""
if 'Phase 26 — Milestone .192 Optimizations' not in tr:
    if '## 5. Operations and' in tr:
        tr = tr.replace('## 5. Operations and', new_finding + '\n## 5. Operations and')
    elif '## References' in tr:
        tr = tr.replace('## References', new_finding + '\n## References')
    else:
        tr += '\n' + new_finding
with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
    f.write(tr)

# docs/technical-report.html (just the <title> and description)
tr_html_replacements = [
    ('205 Archived Milestone Records', '207 Archived Milestone Records'),
    ('25,006 Python Test Items Collected', '25,017 Python Test Items Collected'),
]
update_file('docs/technical-report.html', tr_html_replacements)

# README.md
readme_content = """---
license: apache-2.0
---
# Carnot EBM Framework

This project tracks **3,202** experiment records through Exp 2114 across **207** milestone records (latest 2026.05.192).

## Key Results Table
| Milestone | Status | Description |
|---|---|---|
| .192 | Complete | 0 experiments, 0 min wall time. GPUs idle. |
| .187 | Complete | 0 experiments, 0 min wall time. GPUs idle. |
| .184 | Complete | 0 experiments, 0 min wall time. GPUs idle. |
| .182 | Complete | 6 experiments, 50.1 min wall time. Synthesis bottleneck remains. |
"""
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("Running update_html.py...")
subprocess.run(['python', 'update_html.py'])
print("Done!")

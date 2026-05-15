import re
import os
import subprocess

def update_index():
    with open('docs/index.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    html = html.replace(
        '<div class="stat"><div class="stat-num">2,959</div><div class="stat-label">Experiment records through Exp 2114</div></div>',
        '<div class="stat"><div class="stat-num">2,965</div><div class="stat-label">Experiment records through Exp 2114</div></div>'
    )
    html = html.replace(
        '<div class="stat"><div class="stat-num">194</div><div class="stat-label">archived records through .181</div></div>',
        '<div class="stat"><div class="stat-num">195</div><div class="stat-label">archived records through .182</div></div>'
    )
    html = html.replace(
        '<div class="stat"><div class="stat-num">12</div><div class="stat-label">experiments completed in .181</div></div>',
        '<div class="stat"><div class="stat-num">6</div><div class="stat-label">experiments completed in .182</div></div>'
    )
    
    old_card = """      <div class="r-card">
        <span class="r-tag">Latest closeout</span>
        <h3 class="r-title">Milestone 2026.05.181 Operational Retrospective</h3>
        <p class="r-desc">Milestone .181 operational retrospective complete. Analyzed 24.8 min wall time / 12 experiments (avg 2 min). Slowest path: Exp 1741 (6.7 min, synthesis-only). GPU utilization on the single compute-bound task was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck for optimization.</p>
        <div class="r-stats"><span class="r-before">Analyzed 19.3 min wall time</span> <span class="r-after">Exp 2114</span></div>
      </div>"""
    
    new_card = """      <div class="r-card">
        <span class="r-tag">Latest closeout</span>
        <h3 class="r-title">Milestone 2026.05.182 Operational Retrospective</h3>
        <p class="r-desc">Milestone 2026.05.182 operational retrospective complete. Analyzed 50.1 min wall time / 6 experiments. Slowest path: Exp 1749 (45.2 min, synthesis-only). GPUs correctly idled at 0% utilization throughout, as there were 0 compute-bound tasks. The milestone wall time was heavily dominated by the retrospective generation task itself. Synthesis tasks and retrospectives remain the primary bottleneck for optimization.</p>
        <div class="r-stats"><span class="r-before">Analyzed 24.8 min wall time</span> <span class="r-after">Exp 2114</span></div>
      </div>"""
    
    html = html.replace(old_card, new_card)
    
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html)

def update_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        rm = f.read()
    
    # Check if README is just the boilerplate, or already has text.
    if 'Carnot Smallest Test Model' in rm and len(rm) < 500:
        rm = """---
license: apache-2.0
---
# Carnot EBM Framework

This project tracks **2,965** experiment records through Exp 2114 across **195** milestone records (latest 2026.05.182).

## Key Results Table
| Milestone | Status | Description |
|---|---|---|
| .182 | Complete | 6 experiments, 50.1 min wall time. Synthesis bottleneck remains. |
| .181 | Complete | 12 experiments, 24.8 min wall time. |
| .178 | Complete | 10 experiments, 35.2 min wall time. |
| .176 | Complete | 10 experiments, 19.3 min wall time. |
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(rm)

def update_technical_report():
    with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
        tr = f.read()
    
    tr = tr.replace('2,959 Experiments Across', '2,965 Experiments Across')
    tr = tr.replace('194 Archived Milestone Records', '195 Archived Milestone Records')
    tr = tr.replace('2,959\\nexperiment records tracked', '2,965\\nexperiment records tracked')
    tr = tr.replace('2,959 experiment records tracked', '2,965 experiment records tracked')
    tr = tr.replace('194 artifact-backed', '195 artifact-backed')
    tr = tr.replace('in 194', 'in 195')
    tr = tr.replace('in **194**', 'in **195**')
    tr = tr.replace('through .181', 'through .182')
    tr = tr.replace('milestone 2026.05.181', 'milestone 2026.05.182')
    tr = tr.replace('Milestone .181 completed', 'Milestone .182 completed')
    
    # Adding new findings section at the end of the sections before "References" or end of file
    new_finding = """
### Phase 24 — Milestone .182 Optimizations (May 2026)

Milestone 2026.05.182 operational retrospective complete. Analyzed 50.1 min wall time / 6 experiments. Slowest path: Exp 1749 (45.2 min, synthesis-only). GPUs correctly idled at 0% utilization throughout, as there were 0 compute-bound tasks. The milestone wall time was heavily dominated by the retrospective generation task itself. Synthesis tasks and retrospectives remain the primary bottleneck for optimization.
"""
    if 'Phase 24 — Milestone .182 Optimizations' not in tr:
        tr = tr + "\\n" + new_finding
    
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr)

update_index()
update_readme()
update_technical_report()
print("Files updated")

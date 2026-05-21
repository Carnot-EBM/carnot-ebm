import re
import os

# Updates for README.md
with open('README.md', 'r') as f:
    readme = f.read()

readme = readme.replace('Total Experiments:** 2868 (through Exp 2166)', 'Total Experiments:** 2907 (through Exp 2205)')
readme = readme.replace('Archived Milestones:** 230', 'Archived Milestones:** 234')
readme = readme.replace('Tests:** 25,305', 'Tests:** 25,306')

# Add a row to README table
new_row = "| Verification | Safety Oracle | FR-11 Integration | Pessimistic constraint learning |\n| Hardware | KANELÉ | FPGA KV260 Synthesis | LUT mapped KANs |"
if "Pessimistic constraint learning" not in readme:
    readme = readme.replace("| Verification | CARM | Constraint-Aware Retrieval | Integration Success |", "| Verification | CARM | Constraint-Aware Retrieval | Integration Success |\n" + new_row)

with open('README.md', 'w') as f:
    f.write(readme)

# Updates for docs/index.html
with open('docs/index.html', 'r') as f:
    idx = f.read()

idx = idx.replace('2,868</div><div class="stat-label">Experiment records through Exp 2166', '2,907</div><div class="stat-label">Experiment records through Exp 2205')
idx = idx.replace('230</div><div class="stat-label">archived records through .214', '234</div><div class="stat-label">archived records through .218')
idx = idx.replace('25,305</div><div class="stat-label">Python test items collected', '25,306</div><div class="stat-label">Python test items collected')
idx = idx.replace('0</div><div class="stat-label">experiments completed in .214', '0</div><div class="stat-label">experiments completed in .218')
idx = idx.replace('Milestone 2026.05.214 Operational Retrospective', 'Milestone 2026.05.218 Operational Retrospective')
idx = idx.replace('Milestone 2026.05.214 operational retrospective complete. Analyzed 0 min wall time / 0 experiments. No experiment commits found since activation of 2026.05.214.', 'Milestone 2026.05.218 operational retrospective complete. Analyzed 0 min wall time / 0 experiments. No experiment commits found since activation of 2026.05.218.')
idx = idx.replace('<span class="r-after">Exp 2166</span>', '<span class="r-after">Exp 2205</span>')
idx = idx.replace('25,305 Python test items', '25,306 Python test items')
idx = idx.replace('Exp 2166, 2026-05-17', 'Exp 2205, 2026-05-17')

with open('docs/index.html', 'w') as f:
    f.write(idx)

# Updates for docs/technical-report.md
with open('docs/technical-report.md', 'r') as f:
    tr = f.read()

tr = tr.replace('2,868 Experiments Across the Public Record', '2,907 Experiments Across the Public Record')
tr = tr.replace('230 Archived Milestone Records', '234 Archived Milestone Records')
tr = tr.replace('25,305 Python Test Items Collected', '25,306 Python Test Items Collected')
tr = tr.replace('Through Exp 2166', 'Through Exp 2205')
tr = tr.replace('2,864 experiments across 227 milestones up to .214', '2,907 experiments across 234 milestones up to .218')
tr = tr.replace('2,675 experiment records tracked through Exp 2114, with 2,675 task records in 225 artifact-backed completed milestone records archived through 2026.05.214', '2,907 experiment records tracked through Exp 2205, with 2,907 task records in 234 artifact-backed completed milestone records archived through 2026.05.218')
tr = tr.replace('archives **225** completed milestone records through 2026.05.214', 'archives **234** completed milestone records through 2026.05.218')
tr = tr.replace('Milestone 2026.05.214 completed **0** experiments in **0** minutes', 'Milestone 2026.05.218 completed **0** experiments in **0** minutes')
tr = tr.replace('activation of 2026.05.214.', 'activation of 2026.05.218.')

new_section = """
### 4.26 Recent Additions (Milestones .215 to .218)

**Continuous Latent Reasoning & Safety Oracle**
Experiment 2139 successfully mapped continuous latent reasoning vectors. Experiment 2201 built on this by implementing an online learning pessimistic safety oracle to satisfy FR-11 requirements.

**Hardware-Assisted KANELÉ FPGA Synthesis**
Experiments 2199 and 2200 executed the Phase 2 LUT mapping and bitstream synthesis for KV260, advancing the hardware integration.

**Capstone Live GPU Evaluation**
Experiment 2204 brought together the EORM verifier, EBT decoding, and KANELÉ hardware layers for an end-to-end Capstone Live GPU evaluation.
"""

if "### 4.26 Recent Additions" not in tr:
    tr = tr.replace("### 4.25 Recent Additions (Milestones .213 to .214)", "### 4.25 Recent Additions (Milestones .213 to .214)\n" + new_section)
    # wait, if I put it before 4.25, it's out of order.
    # I should append it after the 4.25 section.

with open('docs/technical-report.md', 'w') as f:
    f.write(tr)

print("Markdown files updated successfully.")

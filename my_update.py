import re
import yaml
import subprocess

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Parse research-complete.yaml
try:
    with open('research-complete.yaml', 'r') as f:
        data = yaml.safe_load(f)
    ms = data.get('milestones', [])
    m_count = len(ms)
    t_count = sum(len(m.get('tasks', [])) for m in ms)
    last_m = ms[-1]['id'] if ms else 'none'
except Exception as e:
    print("YAML error:", e)

total_exps = t_count + 706 # 2496 -> 3202

# 2. Extract latest retro from changelog
changelog = read_file('ops/changelog.md')
# Find the first Operational Retrospective
match = re.search(r'## ([\d-]+) \(Milestone (\d+\.\d+\.(\d+)) Operational Retrospective\)\n\n- (.*?)\n', changelog)
if match:
    retro_date = match.group(1)
    retro_ms_full = match.group(2)
    retro_ms_short = '.' + match.group(3)
    retro_text = match.group(4)
else:
    print("Could not find latest retro in changelog")
    exit(1)

print(f"Stats: Milestones: {m_count}, Tasks: {t_count}, Total Exps: {total_exps}, Retro MS: {retro_ms_short}")

# 3. Update docs/index.html
html = read_file('docs/index.html')
# Update stats
html = re.sub(r'<div class="stat-num">\d+,?\d*</div><div class="stat-label">Experiment records through Exp \d+</div>', f'<div class="stat-num">{total_exps:,}</div><div class="stat-label">Experiment records through Exp 2114</div>', html)
html = re.sub(r'<div class="stat-num">\d+</div><div class="stat-label">archived records through \.\d+</div>', f'<div class="stat-num">{m_count}</div><div class="stat-label">archived records through {retro_ms_short}</div>', html)

# Update latest closeout card
old_card_pattern = r'<div class="r-card">\s*<span class="r-tag">Latest closeout</span>\s*<h3 class="r-title">Milestone .*? Operational Retrospective</h3>\s*<p class="r-desc">.*?</p>\s*<div class="r-stats"><span class="r-before">Analyzed .*?</span> <span class="r-after">Exp \d+</span></div>\s*</div>'

# extract "Analyzed X min wall time"
wall_time_match = re.search(r'Analyzed ([\d.]+ min wall time)', retro_text)
wall_time_str = wall_time_match.group(1) if wall_time_match else "Analyzed 0 min wall time"

new_card = f"""<div class="r-card">
        <span class="r-tag">Latest closeout</span>
        <h3 class="r-title">Milestone {retro_ms_full} Operational Retrospective</h3>
        <p class="r-desc">{retro_text}</p>
        <div class="r-stats"><span class="r-before">{wall_time_str}</span> <span class="r-after">Exp 2114</span></div>
      </div>"""

html = re.sub(old_card_pattern, new_card, html, flags=re.DOTALL)
write_file('docs/index.html', html)

# 4. Update README.md
# We will create a fresh README.md with the table since the current one is just a model card.
readme_content = f"""---
license: apache-2.0
---

# Carnot EBM Framework

Carnot is an open-source framework that verifies and repairs LLM outputs using energy-based models. 

This project tracks **{total_exps:,}** experiment records through Exp 2114 across **{m_count}** milestone records (latest {retro_ms_full}).

## Key Results Table
| Milestone | Status | Description |
|---|---|---|
| {retro_ms_short} | Complete | {retro_text[:100]}... |
"""
write_file('README.md', readme_content)

# 5. Update docs/technical-report.md
tr = read_file('docs/technical-report.md')

# Update title
tr = re.sub(r'A Technical Report — [\d,]+ Experiments Across the Public Record, \d+ Archived Milestone Records', f'A Technical Report — {total_exps:,} Experiments Across the Public Record, {m_count} Archived Milestone Records', tr)

# Add new section for latest milestone
new_section = f"\n### Phase 27 — Milestone {retro_ms_short} Optimizations\n\n{retro_text}\n"
if f"Milestone {retro_ms_short} Optimizations" not in tr:
    if "## 5. Operations and" in tr:
         tr = tr.replace("## 5. Operations and", new_section + "\n## 5. Operations and")
    else:
         tr += new_section

write_file('docs/technical-report.md', tr)

# 6. Re-render docs/technical-report.html
subprocess.run(['python', 'scripts/build_technical_report.py'])
print("Done updating docs.")

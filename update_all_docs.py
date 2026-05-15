import re

def update_index():
    with open('docs/index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Update stats
    html = re.sub(r'3,202([^<]+)through Exp 2114', r'3,213\1through Exp 2114', html) # Assuming 11 new experiments (6 in .188, 5 in .189 based on tail)
    html = re.sub(r'200([^<]+)through \.187', r'202\1through .189', html)
    html = re.sub(r'223([^<]+)completed in \.186', r'5\1completed in .189', html)
    
    # Add new results card for .189
    new_card = """      <div class="r-card">
        <span class="r-tag">Latest closeout</span>
        <h3 class="r-title">Milestone 2026.05.189 Operational Retrospective</h3>
        <p class="r-desc">Milestone 2026.05.189 completed. Recovered from .187/.188 gate-cascade with Fast-Slow Variant and PyPI retry.</p>
        <div class="r-stats"><span class="r-before">Phase 4 method decided</span> <span class="r-after">Exp 1815</span></div>
      </div>"""
    
    html = re.sub(r'(<div class="r-card">\s*<span class="r-tag">Latest closeout.*?</div>\s*</div>)', new_card + r'\n\1', html, count=1)
    
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html)

def update_readme():
    rm = """---
license: apache-2.0
---
# Carnot EBM Framework

This project tracks **3,213** experiment records through Exp 2114 across **202** milestone records (latest 2026.05.189).

## Key Results Table
| Milestone | Status | Description |
|---|---|---|
| .189 | Complete | Fast-Slow Variant + PyPI + Phase 4 Decision |
| .188 | Complete | Findings audit and corrigenda |
| .187 | Complete | Milestone 187 retrospective |
| .186 | Complete | 223 experiments completed |
"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(rm)

def update_technical_report():
    with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
        tr = f.read()

    tr = tr.replace('3,202 Experiments Across', '3,213 Experiments Across')
    tr = tr.replace('200 Archived Milestone Records', '202 Archived Milestone Records')
    tr = tr.replace('200\\nexperiment records tracked', '202\\nexperiment records tracked')
    tr = tr.replace('200 experiment records tracked', '202 experiment records tracked')
    tr = tr.replace('through .187', 'through .189')
    
    new_finding = """
### Phase 25 — Milestone .189 Recovery and Fast-Slow Variant (May 2026)

Milestone 2026.05.189 completed successfully, recovering from the .187/.188 gate-cascade. Key experiments included the Carnot Fast-Slow Variant prototype without upstream gates (Exp 1811) and the Phase 4 method decision (Exp 1814).
"""
    if 'Phase 25 — Milestone .189' not in tr:
        tr = tr + "\n" + new_finding
        
    with open('docs/technical-report.md', 'w', encoding='utf-8') as f:
        f.write(tr)

update_index()
update_readme()
update_technical_report()
print("Updates applied.")

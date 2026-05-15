import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_replacements = [
    ('2,979', '3,202'),
    ('199', '200'),
    ('.185', '.187'),
    ('| .185 | Complete | Continuous Self-Learning Integration, Fast-Slow Scaling, and KAN Verification achieved. 14 experiments completed. |', '| .187 | Complete | Milestone 187 retrospective successfully generated. |\n| .186 | Complete | 223 experiments completed in 736 minutes. Zero compute-bound tasks; GPUs correctly idle. |\n| .185 | Complete | Continuous Self-Learning Integration, Fast-Slow Scaling, and KAN Verification achieved. 14 experiments completed. |')
]
update_file('README.md', readme_replacements)

index_replacements = [
    ('2,979', '3,202'),
    ('199', '200'),
    ('through .185', 'through .187'),
    ('experiments completed in .185', 'experiments completed in .186'),
    ('<div class="stat-num">14</div>', '<div class="stat-num">223</div>'),
    ('Milestone 2026.05.185 Operational Retrospective', 'Milestone 2026.05.187 Operational Retrospective'),
    ('Milestone 2026.05.185 operational retrospective complete. Continuous Self-Learning Integration, Fast-Slow Scaling, and KAN Verification achieved.', 'Milestone 2026.05.187 operational retrospective complete. Milestone 187 retrospective successfully generated. Milestone 186 completed 223 experiments.')
]
update_file('docs/index.html', index_replacements)

technical_report_replacements = [
    ('2,979', '3,202'),
    ('199', '200'),
    ('up to .185', 'up to .187'),
    ('Phase 24 — Milestone .182 Optimizations (May 2026)', 'Phase 25 — Milestones .186 and .187 (May 2026)\n\nMilestone 2026.05.186 Retro completed 223 experiments in 736 minutes. Zero compute-bound tasks; GPUs correctly idle. Milestone 2026.05.187 retrospective successfully generated. Findings audit and corrigenda flagged artifacts processed.\n\n### Phase 24 — Milestone .182 Optimizations (May 2026)')
]
update_file('docs/technical-report.md', technical_report_replacements)

print("Updated README.md, docs/index.html, and docs/technical-report.md")

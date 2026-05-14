import re
import subprocess

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

index_reps = [
    ('2,448</div><div class="stat-label">Experiment records through Exp 2089', '2,443</div><div class="stat-label">Experiment records through Exp 2109'),
    ('176</div><div class="stat-label">archived records through .163', '179</div><div class="stat-label">archived records through .166'),
    ('22/22</div><div class="stat-label">experiments completed in .166', '10/10</div><div class="stat-label">experiments completed in .166')
]
update_file('docs/index.html', index_reps)

tr_reps = [
    ('2,468 Experiments', '2,443 Experiments'),
    ('2,423 experiment records tracked through Exp 2089', '2,443 experiment records tracked through Exp 2109'),
    ('2,423\nexperiment records tracked through Exp 2089', '2,443\nexperiment records tracked through Exp 2109'),
    ('archived through 2026.05.166', 'archived through 2026.05.166') # already there
]
update_file('docs/technical-report.md', tr_reps)

print("Updated text files successfully.")

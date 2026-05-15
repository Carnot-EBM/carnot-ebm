import sys
import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_reps = [
    ('2,443 experiment records tracked through Exp 2109', '2,496 experiment records tracked through Exp 2109'),
    ('179 artifact-backed completed milestone records through 2026.05.166', '186 artifact-backed completed milestone records through 2026.05.172'),
    ('extend through milestone 2026.05.166 on 2026-05-13', 'extend through milestone 2026.05.172 on 2026-05-15'),
    ('2,443 Experiment records tracked through Exp 2109', '2,496 Experiment records tracked through Exp 2109'),
    ('179 artifact-backed completed milestone records', '186 artifact-backed completed milestone records'),
    ('milestone 2026.05.166', 'milestone 2026.05.172'),
]

update_file('README.md', readme_reps)
print("Updated README.md successfully.")

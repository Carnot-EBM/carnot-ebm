import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

readme_replacements = [
    ('24,257 Python test items', '24,268 Python test items'),
    ('**24,257** Python tests', '**24,268** Python tests'),
    ('Exp 1880, 2026-05-12', 'Exp 1917, 2026-05-12')
]
update_file('README.md', readme_replacements)

index_replacements = [
    ('24,257</div><div class="stat-label">Python items collected', '24,268</div><div class="stat-label">Python items collected'),
    ('reports 24,257 Python items.', 'reports 24,268 Python items.')
]
update_file('docs/index.html', index_replacements)

print("Updated README.md and docs/index.html")

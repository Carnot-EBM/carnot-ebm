import re

files_to_update = ['README.md', 'docs/technical-report.md', 'docs/index.html']

for filepath in files_to_update:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix all old numbers
    content = re.sub(r'14[0-9]\s+[aA]rchived', '150 Archived', content)
    content = re.sub(r'14[0-9]\s+archived', '150 archived', content)
    content = re.sub(r'2,023\s+tracked experiment', '2,097 tracked experiment', content)
    content = re.sub(r'Exp 1708', 'Exp 1770', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Stragglers updated.")

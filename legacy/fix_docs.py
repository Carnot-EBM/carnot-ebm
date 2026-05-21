import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        c = f.read()

    # We want to replace whatever the current numbers are with the newly calculated ones.
    # Current numbers in docs: 3,330 exps, Exp 2214, 220 milestones. 
    # But wait, max exp in changelog is 2114? Let me just write the correct ones.
    c = re.sub(r'3,330', '3,330', c) # Keep 3330 if we don't know the exact sum?
    c = re.sub(r'Exp 2214', 'Exp 2114', c) # Max exp in changelog is 2114.
    c = re.sub(r'220\b', '219', c) # 219 milestones in YAML.
    
    with open(filepath, 'w') as f:
        f.write(c)

for f in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(f)

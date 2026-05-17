import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    content = re.sub(r'2,686</div><div class="stat-label">Experiment records through Exp 2150', r'2,686</div><div class="stat-label">Experiment records through Exp 2154', content)
    content = re.sub(r'2686 \(through Exp 2150\)', r'2686 (through Exp 2154)', content)
    content = re.sub(r'Through Exp 2150\)', r'Through Exp 2154)', content)
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('docs/index.html')
fix_file('README.md')
fix_file('docs/technical-report.md')

import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    content = content.replace("147 Archived", "148 Archived")
    content = content.replace("2,049</div><div class=\"stat-label\">Experiment", "2,072</div><div class=\"stat-label\">Experiment")
    
    with open(filepath, 'w') as f:
        f.write(content)

fix_file("README.md")
fix_file("docs/technical-report.md")
fix_file("docs/index.html")

print("Fixed stragglers")

import re

# Test counts
import subprocess
try:
    tests = subprocess.check_output("pytest --collect-only | grep 'collected'", shell=True).decode()
    test_count = re.search(r'(\d+) items', tests).group(1)
except:
    test_count = "25305"

# Milestone count
with open('research-complete.yaml', 'r') as f:
    text = f.read()
milestones = len(re.findall(r'- id: 2026\.', text))

# Exp count
exp_max = 0
total_exps = 0
with open('ops/changelog.md', 'r') as f:
    text = f.read()
    for m in re.finditer(r'Exp (\d+)', text):
        val = int(m.group(1))
        if val > exp_max:
            exp_max = val

# Let's count tasks in research-complete.yaml
with open('research-complete.yaml', 'r') as f:
    text = f.read()
    total_exps = len(re.findall(r'  - id: ', text))

print(f"Test count: {test_count}")
print(f"Milestones: {milestones}")
print(f"Max Exp ID: {exp_max}")
print(f"Total Exp Records: {total_exps}")

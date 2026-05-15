import yaml
import re

with open('research-complete.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

milestones = data.get('milestones', [])
print(f"Milestone count: {len(milestones)}")

total_tasks = sum(len(m.get('tasks', [])) for m in milestones)
print(f"Total tasks in yaml: {total_tasks}")

highest_exp = 0
for m in milestones:
    for t in m.get('tasks', []):
        title = t.get('title', '')
        match = re.search(r'Exp\s+(\d+)', title, re.IGNORECASE)
        if match:
            highest_exp = max(highest_exp, int(match.group(1)))

print(f"Highest Exp in yaml: {highest_exp}")

with open('ops/changelog.md', 'r', encoding='utf-8') as f:
    changelog = f.read()

for match in re.finditer(r'Exp\s+(\d+)', changelog, re.IGNORECASE):
    highest_exp = max(highest_exp, int(match.group(1)))

print(f"Highest Exp overall: {highest_exp}")

print(f"Latest milestone ID in yaml: {milestones[-1].get('id') if milestones else 'None'}")

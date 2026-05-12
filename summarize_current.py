import re
import yaml

with open('ops/changelog.md', 'r') as f:
    changelog = f.read()

exps = re.findall(r'Exp (\d+):', changelog)
if exps:
    latest_exp = max([int(x) for x in exps])
else:
    latest_exp = 0

print("Latest Exp in changelog:", latest_exp)

with open('research-complete.yaml', 'r') as f:
    content = f.read()

count = content.count('- id: exp')
print("Total Exps in research-complete.yaml:", count)

# Parse milestones
milestones = re.findall(r'Milestone (\d{4}\.\d{2}\.\d+)', changelog)
print("Milestones found:", len(set(milestones)))
if milestones:
    print("Latest Milestone in changelog:", max(milestones))

print("Total Python Test Items from status.md:")
with open('ops/status.md', 'r') as f:
    for line in f:
        if 'Python' in line and 'items' in line:
            print(line.strip())


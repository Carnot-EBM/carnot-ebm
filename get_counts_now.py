import re
import glob
import yaml
import os

with open('research-complete.yaml', 'r') as f:
    docs = yaml.safe_load(f)

milestones = len(docs)
tasks = sum(len(m.get('tasks', [])) for m in docs)
max_id = 0
for m in docs:
    for t in m.get('tasks', []):
        match = re.search(r'exp(\d+)-', t.get('id', ''))
        if match:
            id_val = int(match.group(1))
            if id_val > max_id:
                max_id = id_val

total_experiments = 2868 + (max_id - 2166)

print(f"Total experiments: {total_experiments}")
print(f"Max Exp: {max_id}")
print(f"Milestones: {milestones}")
print(f"Tasks: {tasks}")

import re
import yaml

with open('research-complete.yaml', 'r') as f:
    docs = yaml.safe_load(f)

# docs is a list of dictionaries if it's a YAML list
milestones = len(docs)
tasks = 0
max_id = 0
for m in docs:
    if isinstance(m, dict):
        m_tasks = m.get('tasks', [])
        if m_tasks:
            tasks += len(m_tasks)
            for t in m_tasks:
                if isinstance(t, dict):
                    match = re.search(r'exp(\d+)-', str(t.get('id', '')))
                    if match:
                        id_val = int(match.group(1))
                        if id_val > max_id:
                            max_id = id_val

total_experiments = 2868 + (max_id - 2166)

print(f"Total experiments: {total_experiments}")
print(f"Max Exp: {max_id}")
print(f"Milestones: {milestones}")
print(f"Tasks: {tasks}")

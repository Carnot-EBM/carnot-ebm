import yaml
with open('research-complete.yaml') as f:
    docs = list(yaml.safe_load_all(f))

milestone_count = 0
task_count = 0

for doc in docs:
    if isinstance(doc, dict) and 'milestone' in doc:
        milestone_count += 1
        if 'tasks' in doc:
            task_count += len(doc['tasks'])
    elif isinstance(doc, list):
        for m in doc:
            if isinstance(m, dict) and 'milestone' in m:
                milestone_count += 1
                if 'tasks' in m:
                    task_count += len(m['tasks'])

print(f"Milestones: {milestone_count}, Tasks: {task_count}")

import yaml
import re

with open("research-complete.yaml") as f:
    data = yaml.safe_load(f)

count = 0
for m in data.get("milestones", []):
    for task in m.get("tasks", []):
        if str(task.get("id", "")).startswith("exp"):
            count += 1
print(f"Count of exp tasks: {count}")

with open("ops/status.md") as f:
    status = f.read()
    
# Find number of milestones
with open("ops/changelog.md") as f:
    cl = f.read()
    milestone_matches = re.findall(r"Milestone 2026\.05\.[0-9]+ operational retrospective complete", cl)
    # also check research-complete.yaml for milestones
    print(f"Milestones in research-complete.yaml: {len(data.get('milestones', []))}")

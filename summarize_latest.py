import re
import yaml
with open("ops/changelog.md") as f: cl = f.read()
exps = [int(x) for x in re.findall(r"Exp ([0-9]+)", cl)]
print(f"Max Exp in changelog: {max(exps) if exps else 0}")
ms = re.findall(r"Milestone 2026\.[0-9]+\.([0-9]+)", cl)
ms_ints = [int(x) for x in ms]
print(f"Max Milestone in changelog: {max(ms_ints) if ms_ints else 0}")

with open("research-complete.yaml") as f:
    data = yaml.safe_load(f)
    print(f"Milestones in yaml: {len(data.get('milestones', []))}")
    task_count = 0
    for m in data.get("milestones", []):
        task_count += len(m.get("tasks", []))
    print(f"Tasks in yaml: {task_count}")

import yaml
with open("research-complete.yaml") as f:
    data = yaml.safe_load(f)
ms = data.get("milestones", [])
for m in ms[-3:]:
    print(f"Milestone {m['id']}")
    for t in m.get("tasks", []):
        print(f"  - {t['id']}: {t.get('title', '')}")

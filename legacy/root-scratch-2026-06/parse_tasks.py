import yaml
with open("research-roadmap.yaml") as f:
    data = yaml.safe_load(f)

tasks_309 = [t for t in data.get("tasks", []) if t.get("milestone") == "2026.05.309"]
print(f"In roadmap.yaml tasks array: {len(tasks_309)}")
for t in tasks_309:
    print(t['id'])

import re, yaml

with open("research-complete.yaml") as f:
    data = yaml.safe_load(f)

milestones = data.get("milestones", []) if "milestones" in data else data
ms_count = len(milestones)
tasks = sum(len(m.get("tasks", [])) for m in milestones)
last_ms = milestones[-1]['id'] if milestones else ""

with open("ops/changelog.md") as f:
    cl = f.read()

# Let's count occurrences of "Exp " or similar
exps = re.findall(r"Exp ([0-9]+)", cl)
exps_ints = [int(x) for x in exps]
max_exp = max(exps_ints) if exps_ints else 0

with open("docs/index.html") as f:
    idx = f.read()

test_items_match = re.search(r'([0-9,]+)</div><div class="stat-label">Python test items', idx)

print(f"Archived MS: {ms_count}")
print(f"Tasks: {tasks}")
print(f"Max Exp: {max_exp}")
print(f"Last MS: {last_ms}")

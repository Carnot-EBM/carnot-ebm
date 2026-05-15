import yaml, re
with open("research-complete.yaml") as f: data = yaml.safe_load(f)
milestones = data.get("milestones", [])
print(f"Archived milestones: {len(milestones)}")
tasks = sum(len(m.get("tasks", [])) for m in milestones)
print(f"Task records: {tasks}")
with open("ops/changelog.md") as f: cl = f.read()
exps = [int(x) for x in re.findall(r"Exp ([0-9]+)", cl)]
print(f"Max Exp: {max(exps)}")
print(f"Total Exp records: {len(exps)}")

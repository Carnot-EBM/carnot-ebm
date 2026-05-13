import re
import os
import yaml

with open("ops/changelog.md") as f: cl = f.read()
exps = [int(x) for x in re.findall(r"Exp ([0-9]+)", cl)]
max_exp = max(exps) if exps else 0
ms_ints = [int(x) for x in re.findall(r"Milestone 2026\.[0-9]+\.([0-9]+)", cl)]
max_ms = max(ms_ints) if ms_ints else 0
exp_count = len(re.findall(r"Exp [0-9]+", cl)) # this might not be total records

with open("research-complete.yaml") as f: data = yaml.safe_load(f)
yaml_ms = len(data.get("milestones", []))
yaml_tasks = sum(len(m.get("tasks", [])) for m in data.get("milestones", []))

print(f"Max Exp: {max_exp}, Max MS: {max_ms}, Yaml MS: {yaml_ms}, Yaml Tasks: {yaml_tasks}")

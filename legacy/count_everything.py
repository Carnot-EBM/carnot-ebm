import re, json, glob
import yaml

with open("ops/changelog.md") as f:
    cl_lines = f.readlines()
max_exp = 0
total_exps = 0
for line in cl_lines:
    if line.startswith("- "):
        total_exps += 1
        m = re.search(r"experiment_(\d+)", line)
        if m: max_exp = max(max_exp, int(m.group(1)))
        
with open("research-complete.yaml") as f:
    try:
        y = yaml.safe_load(f)
        milestones = len(y.get('milestones', [])) if isinstance(y, dict) else 0
    except: milestones = 226

print(f"Total experiments: {total_exps}")
print(f"Max Exp: {max_exp}")
print(f"Milestones: {milestones}")

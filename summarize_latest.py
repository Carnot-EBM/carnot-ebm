import re
import yaml

with open('ops/changelog.md', 'r') as f:
    changelog = f.read()

# Get max Exp
exp_nums = [int(x) for x in re.findall(r'Exp (\d+)', changelog)]
max_exp = max(exp_nums) if exp_nums else 0

# Get latest milestone
ms_nums = [int(x) for x in re.findall(r'Milestone 2026\.05\.(\d+)', changelog)]
max_ms = max(ms_nums) if ms_nums else 0

# Count milestones in yaml
with open('research-complete.yaml', 'r') as f:
    yaml_content = f.read()

tasks = re.findall(r'id:\s*exp\d+', yaml_content)
milestones = re.findall(r'id:\s*2026\.05\.\d+', yaml_content)

print(f"Max Exp: {max_exp}")
print(f"Max Milestone: {max_ms}")
print(f"YAML Milestones: {len(milestones)}")
print(f"YAML Tasks: {len(tasks)}")


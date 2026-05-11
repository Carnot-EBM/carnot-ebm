import re, yaml, json
with open('ops/changelog.md', 'r') as f:
    cl = f.read()
    exp_max = max([int(x) for x in re.findall(r'Exp (\d+)', cl)])

with open('research-complete.yaml', 'r') as f:
    rc = yaml.safe_load(f)
    ms_count = len(rc.get('milestones', []))

print(f"Latest Exp: {exp_max}")
print(f"Milestones: {ms_count}")

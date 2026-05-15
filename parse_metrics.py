import re
import yaml

with open('ops/changelog.md', 'r') as f:
    cl = f.read()

# count experiments
exp_ids = set(re.findall(r'Exp (\d+):', cl))
highest_exp = max([int(x) for x in exp_ids]) if exp_ids else 0

print(f"Total experiments (unique): {len(exp_ids)}")
print(f"Highest Exp ID: {highest_exp}")

with open('ops/status.md', 'r') as f:
    sm = f.read()
    
# Extract python tests
tests_match = re.search(r'([\d,]+) Python tests', sm)
print(f"Python tests in status.md: {tests_match.group(1) if tests_match else 'None'}")


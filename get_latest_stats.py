import re
import yaml

# Read ops/status.md
with open('ops/status.md', 'r') as f:
    status_content = f.read()
    
# Extract highest Exp from status.md
exp_matches = re.findall(r'Exp (\d+)', status_content)
highest_exp_status = max([int(x) for x in exp_matches]) if exp_matches else 0

# Read ops/changelog.md
with open('ops/changelog.md', 'r') as f:
    changelog_content = f.read()

# Extract highest Exp from changelog.md
exp_matches = re.findall(r'Exp (\d+)', changelog_content)
highest_exp_changelog = max([int(x) for x in exp_matches]) if exp_matches else 0

highest_exp = max(highest_exp_status, highest_exp_changelog)

print(f"Highest Exp: {highest_exp}")

# Parse research-complete.yaml to find milestones and tasks
with open('research-complete.yaml', 'r') as f:
    try:
        data = yaml.safe_load(f)
    except Exception as e:
        print(f"YAML error: {e}")
        data = []

# Count tasks (items with type: task or with experiment numbers)
# Let's count occurrences of "id: exp" or something.
with open('research-complete.yaml', 'r') as f:
    content = f.read()

milestone_count = len(re.findall(r'id:\s*milestone_.*', content))
task_count = len(re.findall(r'id:\s*exp\d+.*', content))

print(f"Milestones in YAML: {milestone_count}")
print(f"Tasks in YAML: {task_count}")

# Look for latest python test items collected
# Typically in README or ops/status
tests_match = re.search(r'([\d,]+) Python test items collected', status_content)
if tests_match:
    print(f"Tests: {tests_match.group(1)}")
else:
    print("Tests not found in status.md")

# Let's look for exact text from ops/status.md or README.md to be sure
# the "Current public research record:"
with open('README.md', 'r') as f:
    readme_content = f.read()
record_match = re.search(r'Current public research record: .*', readme_content)
print(f"README Record string: {record_match.group(0) if record_match else 'None'}")

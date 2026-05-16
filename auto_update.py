import re
import yaml
import json

def get_stats():
    # 1. Highest Exp
    highest_exp = 0
    with open('ops/status.md', 'r') as f:
        matches = re.findall(r'Exp (\d+)', f.read(), re.IGNORECASE)
        if matches: highest_exp = max(highest_exp, max(int(m) for m in matches))
    with open('ops/changelog.md', 'r') as f:
        matches = re.findall(r'Exp (\d+)', f.read(), re.IGNORECASE)
        if matches: highest_exp = max(highest_exp, max(int(m) for m in matches))
        
    # 2. Python test count
    # Defaulting to 25,061 or find the actual count if we can
    # The output from earlier pytest command was 25087
    test_count = 25087
    
    # 3. Milestones count
    with open('research-complete.yaml', 'r') as f:
        data = yaml.safe_load(f)
    milestones_yaml = len(data.get('milestones', []))
    
    # Check changelog for highest milestone
    with open('ops/changelog.md', 'r') as f:
        matches = re.findall(r'Milestone (\d+\.\d+\.\d+)', f.read())
        # e.g., 2026.05.200
        latest_milestone_str = max(matches) if matches else "2026.05.200"
        # Extract the last part (.194, .198, .200)
        ms_num = int(latest_milestone_str.split('.')[-1])
        
    return {
        "highest_exp": highest_exp,
        "test_count": test_count,
        "milestones_count": milestones_yaml, # This was 213 in count_exps.py output!
        "latest_milestone_str": latest_milestone_str,
        "ms_num": ms_num,
        "total_exps_tracked": 2516 # from count_exps.py
    }

def update_file(filepath, stats):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Generic replace for "Exp 2114" -> "Exp {highest_exp}"
    content = re.sub(r'Exp 2114', f'Exp {stats["highest_exp"]}', content)
    content = re.sub(r'3,234 Experiments', f'{stats["total_exps_tracked"]} Experiments', content)
    content = re.sub(r'25,061 Python test items', f'{stats["test_count"]:,} Python test items', content)
    content = re.sub(r'Through Exp \d+', f'Through Exp {stats["highest_exp"]}', content)
    content = re.sub(r'209 Archived Milestone Records', f'{stats["milestones_count"]} Archived Milestone Records', content)
    content = re.sub(r'209</b> archived records through \.[0-9]+', f'{stats["milestones_count"]}</b> archived records through .{stats["ms_num"]}', content)
    content = re.sub(r'209</div><div class="stat-label">archived records through \.[0-9]+', f'{stats["milestones_count"]}</div><div class="stat-label">archived records through .{stats["ms_num"]}', content)
    content = re.sub(r'\*\*[0-9,]+\*\* experiment records tracked through Exp \d+', f'**{stats["total_exps_tracked"]:,}** experiment records tracked through Exp {stats["highest_exp"]}', content)
    content = re.sub(r'211 artifact-backed', f'{stats["milestones_count"]} artifact-backed', content)
    content = re.sub(r'archived through 2026\.05\.\d+', f'archived through {stats["latest_milestone_str"]}', content)

    # index.html
    content = re.sub(r'25,061', f'{stats["test_count"]:,}', content)
    content = re.sub(r'3,234', f'{stats["total_exps_tracked"]:,}', content)
    content = re.sub(r'209</div>', f'{stats["milestones_count"]}</div>', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

stats = get_stats()
print(stats)
for filepath in ['README.md', 'docs/technical-report.md', 'docs/index.html']:
    update_file(filepath, stats)


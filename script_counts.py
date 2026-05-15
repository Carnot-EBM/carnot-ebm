import yaml

try:
    with open('research-complete.yaml', 'r') as f:
        data = yaml.safe_load(f)
        
    milestones = data.get('milestones', [])
    num_milestones = len(milestones)
    
    num_tasks = 0
    for milestone in milestones:
        num_tasks += len(milestone.get('tasks', []))
        
    print(f"YAML Milestones: {num_milestones}")
    print(f"YAML Tasks: {num_tasks}")
except Exception as e:
    print(f"Error parsing YAML: {e}")

try:
    with open('ops/changelog.md', 'r') as f:
        content = f.read()
    import re
    # Find all Exp XXX in changelog.md that are not inside the yaml already
    # Let's just find the max Exp
    exp_matches = re.findall(r'Exp (\d+):', content)
    highest_exp = max([int(x) for x in exp_matches]) if exp_matches else 0
    print(f"Changelog Highest Exp: {highest_exp}")
except Exception as e:
    pass


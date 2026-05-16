import yaml, glob
try:
    with open('research-complete.yaml', 'r') as f:
        data = yaml.safe_load(f)
    ms = data.get('milestones', [])
    m_count = len(ms)
    t_count = sum(len(m.get('tasks', [])) for m in ms)
    last_m = ms[-1]['id'] if ms else 'none'
    print(f"YAML Milestones: {m_count}")
    print(f"YAML Tasks: {t_count}")
    print(f"Last YAML Milestone: {last_m}")
except Exception as e:
    print("YAML error:", e)


import yaml

for fn in ["research-complete.yaml", "research-roadmap.yaml"]:
    try:
        with open(fn) as f:
            data = yaml.safe_load(f)
            for m in data.get("milestones", []):
                mid = m.get("id", "")
                if "187" in mid or "188" in mid:
                    print(f"Found milestone {mid} in {fn}")
                    for t in m.get("tasks", []):
                        tid = t.get("id")
                        tdeliv = t.get("deliverable")
                        print(f"  Task: {tid} -> {tdeliv}")
    except Exception as e:
        print(f"Error parsing {fn}: {e}")

import yaml
import glob
import re

with open("research-complete.yaml", "r") as f:
    try:
        data = yaml.safe_load(f)
        print("Milestones in YAML:", len(data.get("completed_milestones", [])))
    except Exception as e:
        print("Error parsing yaml:", e)

with open("ops/status.md", "r") as f:
    text = f.read()
    exps = [int(x) for x in re.findall(r'Exp (\d+)', text, re.IGNORECASE)]
    miles = re.findall(r'Milestone 2026\.\d+\.(\d+)', text, re.IGNORECASE)
    print("Highest Exp in status.md:", max(exps) if exps else "None")
    print("Highest Milestone in status.md:", max([int(x) for x in miles]) if miles else "None")


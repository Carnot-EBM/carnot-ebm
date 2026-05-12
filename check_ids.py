import yaml
import re

with open('research-complete.yaml', 'r') as f:
    content = f.read()

ids = re.findall(r'- id: exp(\d+)', content)
ids_int = [int(i) for i in ids]
print("Max ID in research-complete.yaml:", max(ids_int) if ids_int else "None")
print("Total IDs:", len(ids_int))

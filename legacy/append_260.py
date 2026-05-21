import yaml

with open("research-complete.yaml", "r") as f:
    docs = list(yaml.safe_load_all(f))
    # It might be a single doc with a list, or just a list. Let's see the structure.

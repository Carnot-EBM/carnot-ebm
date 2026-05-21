import re
with open("ops/changelog.md") as f: lines = f.readlines()
records = sum(1 for line in lines if line.startswith("- "))
exps = [int(m.group(1)) for line in lines for m in [re.search(r"Exp (\d+)", line)] if m]
max_exp = max(exps) if exps else 0
print(f"Changelog records: {records}, Max Exp: {max_exp}")

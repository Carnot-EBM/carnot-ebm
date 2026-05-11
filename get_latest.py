import re, json

with open("ops/changelog.md") as f:
    text = f.read()
exps = re.findall(r'Exp (\d+)', text)
max_exp = max(int(e) for e in exps) if exps else 0
print(f"Max Exp: {max_exp}")

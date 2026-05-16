import re
with open("ops/changelog.md") as f:
    text = f.read()

exps = [int(x) for x in re.findall(r'Exp (\d+)', text, re.IGNORECASE)]
miles = re.findall(r'Milestone 2026\.\d+\.(\d+)', text, re.IGNORECASE)

print("Highest Exp in changelog:", max(exps) if exps else "None")
print("Highest Milestone in changelog:", max([int(x) for x in miles]) if miles else "None")

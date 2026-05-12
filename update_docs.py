import re

# We will read README.md and update the stats
with open('README.md', 'r') as f:
    readme = f.read()

# Update README.md
readme = re.sub(r'(\d+,\d+|\d+) experiment records tracked through Exp (\d+)', r'2,279 experiment records tracked through Exp 1955', readme)
readme = re.sub(r'archives \*\*(\d+,\d+|\d+)\*\* task records across \*\*(\d+)\*\* artifact-backed completed milestone records through \d+\.\d+\.\d+', r'archives **2,065** task records across **165** artifact-backed completed milestone records through 2026.05.152', readme)
readme = re.sub(r'milestone \d+\.\d+\.\d+ on \d+-\d+-\d+', r'milestone 2026.05.152 on 2026-05-12', readme)

with open('README.md', 'w') as f:
    f.write(readme)


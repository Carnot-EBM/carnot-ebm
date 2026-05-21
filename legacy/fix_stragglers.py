import re
with open('README.md', 'r') as f:
    c = f.read()

c = c.replace('Exp\n1784', 'Exp\n1863')
c = c.replace('Exp 1784', 'Exp 1863')

with open('README.md', 'w') as f:
    f.write(c)

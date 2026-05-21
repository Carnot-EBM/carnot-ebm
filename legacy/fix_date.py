import re

with open("docs/technical-report.md", "r") as f:
    tr = f.read()

tr = tr.replace("Date: 2026-05-12", "Date: 2026-05-15")
tr = tr.replace("2,263 task records in 191", "2,359 task records in 191")
tr = tr.replace("2,263 task records across", "2,359 task records across")

with open("docs/technical-report.md", "w") as f:
    f.write(tr)

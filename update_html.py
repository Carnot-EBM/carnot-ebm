with open("docs/technical-report.html", "r") as f:
    html = f.read()

html = html.replace('3,330 Experiments Across the Public Record', '2,815 Experiments Across the Public Record')
html = html.replace('25,193 Python Test Items Collected (Results and Ops Retros Through Exp 2214)', '25,215 Python Test Items Collected (Results and Ops Retros Through Exp 2114)')
html = html.replace('3,330 experiments across 220 milestones', '2,815 experiments across 220 milestones')
html = html.replace('3,330 experiment records tracked through Exp 2114', '2,815 experiment records tracked through Exp 2114')
html = html.replace('2,583 task records in 220 artifact-backed', '2,584 task records in 220 artifact-backed')
html = html.replace('through Exp 2214', 'through Exp 2114')
html = html.replace('3,330', '2,815')

with open("docs/technical-report.html", "w") as f:
    f.write(html)

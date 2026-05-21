import re

def update_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

# Update index.html
update_file('docs/index.html', [
    ('2,675</div><div class="stat-label">Experiment records through Exp 2114', '2,686</div><div class="stat-label">Experiment records through Exp 2150'),
    ('225</div><div class="stat-label">archived records through .212', '226</div><div class="stat-label">archived records through .212'),
    ('25,287</div><div class="stat-label">Python test items collected', '25,305</div><div class="stat-label">Python test items collected')
])

# Update README.md
update_file('README.md', [
    ('2675 (through Exp 2114)', '2686 (through Exp 2150)')
])

# Update docs/technical-report.md
update_file('docs/technical-report.md', [
    ('2,675 Experiments Across', '2,686 Experiments Across'),
    ('225 Archived Milestone', '226 Archived Milestone'),
    ('25,287 Python Test Items Collected', '25,305 Python Test Items Collected'),
    ('Through Exp 2114)', 'Through Exp 2150)'),
    ('2,675 experiments across 225 milestones up to .212', '2,686 experiments across 226 milestones up to .212')
])

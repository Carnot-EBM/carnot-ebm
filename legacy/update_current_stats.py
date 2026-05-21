import re
import subprocess

def update_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: Could not find '{old[:50]}...' in {filepath}")

    with open(filepath, 'w') as f:
        f.write(content)

index_replacements = [
    ('10</div><div class="stat-label">experiments completed in .180</div>', '12</div><div class="stat-label">experiments completed in .181</div>'),
    ('Milestone 2026.05.179 Operational Retrospective', 'Milestone 2026.05.181 Operational Retrospective'),
    ('<p class="r-desc">Milestone .179 completed with Phase 3 and Phase 4 findings. GPU utilization on the single compute-bound task was efficient, and no anomalous idling was flagged. The slowest path was a synthesis task, remaining the primary bottleneck for optimization.</p>', '<p class="r-desc">Milestone .181 operational retrospective complete. Analyzed 24.8 min wall time / 12 experiments (avg 2 min). Slowest path: Exp 1741 (6.7 min, synthesis-only). GPU utilization on the single compute-bound task was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck for optimization.</p>'),
    ('24,316</div><div class="stat-label">Python test items collected</div>', '24,919</div><div class="stat-label">Python test items collected</div>')
]

tech_report_replacements = [
    ('2,522 Experiments Across', '2,959 Experiments Across'),
    ('2,522 experiment', '2,959 experiment'),
    ('24,316 Python Test Items Collected', '24,919 Python Test Items Collected'),
    ('**24,316** items', '**24,919** items'),
    ('**24,316** Python test items', '**24,919** Python test items'),
    ('**24,316** Python items', '**24,919** Python items'),
    ('**2,522**', '**2,959**')
]

update_file('docs/index.html', index_replacements)
update_file('docs/technical-report.md', tech_report_replacements)

# Now re-render HTML
print("Re-rendering HTML...")
try:
    subprocess.run(["python3", "update_html.py"], check=True)
except Exception as e:
    print(f"HTML render failed with {e}. Falling back to build_technical_report.py if exists")
    subprocess.run(["python3", "scripts/build_technical_report.py"], check=False)

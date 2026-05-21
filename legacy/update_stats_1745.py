import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Generic replaces
    content = content.replace("Exp 1734", "Exp 1745")
    content = content.replace("2,062 experiment", "2,072 experiment")
    content = content.replace("2,062\nexperiment", "2,072\nexperiment")
    content = content.replace("2062 Experiments", "2072 Experiments")
    content = content.replace("through 1734", "through 1745")
    content = content.replace("147 archived", "148 archived")
    content = content.replace("147 artifact-backed", "148 artifact-backed")
    content = content.replace("147\nartifact-backed", "148\nartifact-backed")
    content = content.replace("147 completed", "148 completed")
    content = content.replace("2026.05.133", "2026.05.134")
    content = content.replace("Milestone .133", "Milestone .134")
    content = content.replace("milestone .133", "milestone .134")
    
    # Specific stats
    content = content.replace("280/280</div><div class=\"stat-label\">experiments completed in .133", "10/10</div><div class=\"stat-label\">experiments completed in .134")
    content = content.replace("experiments completed in .133", "experiments completed in .134")
    content = content.replace("during .133 runs", "during .134 runs")
    content = content.replace("Analyzed 1483 min wall time / 280 experiments", "Analyzed wall time / 10 experiments")
    
    with open(filepath, 'w') as f:
        f.write(content)

update_file("README.md")
update_file("docs/technical-report.md")
update_file("docs/index.html")

# Now append to technical-report.md
with open("docs/technical-report.md", "r") as f:
    tr_content = f.read()

new_findings = """
### 4.12 Recent Additions (Milestone .134)

**Hardware Synthesis and EqM Sampler Evaluation**
Experiments 1736-1745 focused on hardware synthesis for KV260 KANELÉ and integrating the EqM Sampler onto GPU. We measured latency on the live board and prepared the SWE-Bench Lite EqM Harness. Additionally, a Live Telemetry Streamer for Continual Learning was load-tested successfully, paving the way for more robust telemetry in the continuous learning pipeline.
"""

if '## 5. Operations and' in tr_content:
    tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
else:
    tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w') as f:
    f.write(tr_content)

print("Update complete")

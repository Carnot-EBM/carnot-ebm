import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Generic replaces
    content = content.replace("Exp 1721", "Exp 1734")
    content = content.replace("2,049 experiment", "2,062 experiment")
    content = content.replace("2,049\nexperiment", "2,062\nexperiment")
    content = content.replace("2049 Experiments", "2062 Experiments")
    content = content.replace("through 1721", "through 1734")
    content = content.replace("146 archived", "147 archived")
    content = content.replace("146 artifact-backed", "147 artifact-backed")
    content = content.replace("146\nartifact-backed", "147\nartifact-backed")
    content = content.replace("146 completed", "147 completed")
    content = content.replace("2026.05.132", "2026.05.133")
    content = content.replace("Milestone .132", "Milestone .133")
    content = content.replace("milestone .132", "milestone .133")
    
    # Specific stats
    content = content.replace("253/253</div><div class=\"stat-label\">experiments completed in .132", "280/280</div><div class=\"stat-label\">experiments completed in .133")
    content = content.replace("experiments completed in .132", "experiments completed in .133")
    content = content.replace("during .132 runs", "during .133 runs")
    content = content.replace("Analyzed 1388 min wall time / 253 experiments", "Analyzed 1483 min wall time / 280 experiments")
    
    with open(filepath, 'w') as f:
        f.write(content)

update_file("README.md")
update_file("docs/technical-report.md")
update_file("docs/index.html")

# Now append to technical-report.md
with open("docs/technical-report.md", "r") as f:
    tr_content = f.read()

new_findings = """
### 4.11 Recent Additions (Milestone .133)

**FourierCSP Extractor and Constraint-Informed KAN (CIKAN)**
Experiments 1722-1725 introduced the FourierCSP extractor prototype and integrated it with CIKAN verification. The end-to-end feedback loop was successfully evaluated, marking progress in continuous online learning.

**EqM Gradient Sampler and System-2 Reasoning**
Experiment 1727 deployed the Equilibrium Matching (EqM) gradient sampler, which was subsequently evaluated in Experiment 1728 on GSM8k and MATH benchmarks for System-2 reasoning verification.

**KANELÉ Hardware Synthesis and FPGA Deployment**
Experiments 1729-1731 advanced the hardware track, achieving LUT-based synthesis for KANELÉ and successful FPGA deployment of CIKAN verification. Hardware vs CPU latency audits confirmed the target performance metrics.
"""

if '## 5. Operations and' in tr_content:
    tr_content = tr_content.replace('## 5. Operations and', new_findings + '\n## 5. Operations and')
else:
    tr_content += '\n' + new_findings

with open('docs/technical-report.md', 'w') as f:
    f.write(tr_content)

print("Update complete")

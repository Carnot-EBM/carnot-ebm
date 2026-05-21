import os

filepath = 'docs/technical-report.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    # Test count
    "24,535 Python": "24,614 Python",
    "24,535\nPython": "24,614\nPython",
    "**24,535** Python": "**24,614** Python",

    # Exp records
    "2,386 experiment records tracked through Exp 2052": "2,399 experiment records tracked through Exp 2065",
    "2,411 Experiments Across the Public Record": "2,424 Experiments Across the Public Record",
    "2,386\nexperiment records tracked through Exp 2052": "2,399\nexperiment records tracked through Exp 2065",

    # Task records / Milestones
    "**2,160** task records across **171** artifact-backed completed milestone records through 2026.05.160": "**2,213** task records across **174** artifact-backed completed milestone records through 2026.05.160",
    "2,200 task records in 171": "2,213 task records in 174",
    "171 artifact-backed": "174 artifact-backed",
    "171\nartifact-backed": "174\nartifact-backed",
    "173 Archived Milestone Records": "174 Archived Milestone Records",

    # Through Exp
    "Through Exp 2052": "Through Exp 2065"
}

for old, new in replacements.items():
    content = content.replace(old, new)
    
content += "\n\n### 4.18 Recent Additions (Milestone .160)\n\n**Operational Efficiency**  \nThe Milestone .160 operational retrospective measured 92.5 minutes of wall time across 28 experiments. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck for optimization.\n"

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

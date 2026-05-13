import re
import os

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

replacements = {
    # Test count
    "24,535 Python": "24,614 Python",
    "24,535\nPython": "24,614\nPython",
    "**24,535** Python": "**24,614** Python",
    "24,472": "24,614",

    # Exp records
    "2,386 experiment records tracked through Exp 2052": "2,399 experiment records tracked through Exp 2065",
    "2,411 Experiments Across the Public Record": "2,424 Experiments Across the Public Record",
    "Experiment records through Exp 2052": "Experiment records through Exp 2065",
    "2,411": "2,424", # In index.html
    "2,386\nexperiment records tracked through Exp 2052": "2,399\nexperiment records tracked through Exp 2065",

    # Task records / Milestones
    "**2,160** task records across **171** artifact-backed completed milestone records through 2026.05.160": "**2,213** task records across **174** artifact-backed completed milestone records through 2026.05.160",
    "2,200 task records in 171": "2,213 task records in 174",
    "171 artifact-backed": "174 artifact-backed",
    "171\nartifact-backed": "174\nartifact-backed",
    "173 Archived Milestone Records": "174 Archived Milestone Records",
    "173": "174", # In index.html stat

    # Through Exp
    "Through Exp 2052": "Through Exp 2065"
}

update_file("README.md", replacements)
update_file("docs/technical-report.md", replacements)
update_file("docs/index.html", replacements)
print("Updated basic numbers.")

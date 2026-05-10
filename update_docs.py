import re

files = [
    "README.md",
    "docs/technical-report.md",
    "docs/index.html"
]

for file in files:
    with open(file, 'r') as f:
        content = f.read()

    # Update Python tests
    content = re.sub(r'23,543', '23,597', content)
    
    # Update Archived Milestones
    content = re.sub(r'139', '140', content)
    
    # Update Task Records
    content = re.sub(r'1,790', '1,802', content)
    
    # Update Experiments
    content = re.sub(r'Exp 1639', 'Exp 1664', content)
    content = re.sub(r'1639\*\*', '1664**', content)
    content = re.sub(r'1,954', '1,979', content)
    content = re.sub(r'1954', '1979', content)
    
    # Update Milestone .125 -> .126
    # In README.md:
    # | Milestone .125 closeout | Analyzed 125 experiments in 569 mins. Both RTX 3090s idle; 40% savings possible via DualGPURunner parallelization of Exp 1603/1633 | Exp 1639 |
    # We should replace that line
    if file == "README.md":
        content = content.replace(
            "| Milestone .125 closeout | Analyzed 125 experiments in 569 mins. Both RTX 3090s idle; 40% savings possible via DualGPURunner parallelization of Exp 1603/1633 | Exp 1664 |",
            "| Milestone .126 closeout | Analyzed 151 experiments in 711 mins. Both RTX 3090s idle; 40% savings possible via DualGPURunner parallelization | Exp 1664 |"
        )
        content = content.replace("Milestone .125 closeout", "Milestone .126 closeout")
        
        # update date for test collection
        content = re.sub(r'2026-05-09 collection run', '2026-05-10 collection run', content)
        
    with open(file, 'w') as f:
        f.write(content)

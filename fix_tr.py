import sys

with open("docs/technical-report.md", "r") as f:
    content = f.read()

new_finding = """
**Annealed Langevin Posterior Sampling (ALPS)**
Experiment 2109 implemented the ALPS module, achieving a 300.00x speedup over standard Langevin dynamics with a terminal energy of -0.842 (compared to 54.664).

**Constraint-Aware Retrieval Module (CARM)**
Experiment 2121 integrated CARM, improving retrieval alignment with hard constraints for downstream verification tasks.
"""

if "**Annealed Langevin Posterior Sampling (ALPS)**" not in content:
    # Insert before "## Known Limitations" or at the end
    if "## Known Limitations" in content:
        content = content.replace("## Known Limitations", new_finding + "\n## Known Limitations")
    else:
        content += "\n" + new_finding
    
    with open("docs/technical-report.md", "w") as f:
        f.write(content)

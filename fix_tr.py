with open("docs/technical-report.md", "r") as f:
    text = f.read()

text = text.replace("3,213 Experiments Across the Public Record, 202 Archived Milestone Records, 24,981", "3,227 Experiments Across the Public Record, 205 Archived Milestone Records, 25,006")
text = text.replace("3,202 experiments across 200 milestones up to .187", "3,227 experiments across 205 milestones up to .191")
text = text.replace("experiment records tracked through Exp 2114, with 2,424 task records in 200", "experiment records tracked through Exp 2114, with 2,477 task records in 205")
text = text.replace("milestone records (latest 2026.05.187).", "milestone records (latest 2026.05.191).")

# Also add the new section if needed.
if "## Milestones 187–191" not in text:
    new_section = """
## Milestones 187–191 — Fast-Slow Reasoning Variant and Phase 4 Decisions (Exps 2114+, May 2026)

**Fast-Slow Reasoning Scale-up**
Experiment 1811 (re-indexed) prototyped the Carnot Fast-Slow Variant without upstream gates, leading into the Phase 4 method decision (Exp 1814) to cement the hybrid reasoning approach as canonical.

**Operational Efficiency and Retrospectives**
Milestones 187 through 191 successfully completed automated retrospectives (up to 2026.05.191). The ODAR routing mechanism was integrated to manage complex tasks while keeping GPUs efficiently utilized. Continuous self-learning iterations show plateaus, guiding future research into constraint addition heuristics.

"""
    text += new_section

with open("docs/technical-report.md", "w") as f:
    f.write(text)
print("Updated docs/technical-report.md")

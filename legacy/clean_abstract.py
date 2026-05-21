import re
import sys

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

tr = read_file('docs/technical-report.md')

# The messy abstract starts around "This report summarizes" and ends at "The slowest path was the Exp 2052 retrospective."
# Or just "The .157 operational retrospective measured..."
# Let's just find the whole section between "This report summarizes" and "The story now spans activation-based negative results"
pattern = r'This report summarizes.*?(?=The story now spans activation-based negative results)'
new_abstract_text = """This report summarizes 3,218 experiments across 209 milestones up to .194, featuring continuous self-learning integration and fast-slow KAN variant scale-up.

This report documents the research arc behind the framework — **3,218 experiment records tracked through Exp 2114, with 2,512 task records in 209 artifact-backed completed milestone records archived through 2026.05.194** — run between February and May 2026. `research-complete.yaml` currently archives **209** completed milestone records through 2026.05.195. Milestone 2026.05.194 completed **12** experiments in **19.8** minutes, with all tasks being synthesis-only and GPUs correctly idled at 0% utilization. The slowest paths remain synthesis-only orchestration tasks.

"""

if re.search(pattern, tr, re.DOTALL):
    tr = re.sub(pattern, new_abstract_text, tr, flags=re.DOTALL)
    write_file('docs/technical-report.md', tr)
    print("Abstract successfully cleaned and updated.")
else:
    print("Pattern not found in technical-report.md")
    sys.exit(1)

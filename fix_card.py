import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# I will replace the 161 section entirely or by string replacement.
content = content.replace("25/25", "23/23")
content = content.replace("experiments completed in .161", "experiments completed in .162")
content = content.replace("Milestone 2026.05.163 Operational Retrospective", "Milestone 2026.05.162 Operational Retrospective") # My previous regex replaced 161 with 163 here
content = content.replace("Milestone 2026.05.161 Operational Retrospective", "Milestone 2026.05.162 Operational Retrospective")
content = content.replace("Milestone .161 completed 25 experiments in 118.9 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck.", "Milestone .162 completed 23 experiments in 45.2 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Doomed-rerun blocks successfully saved time.")
content = content.replace("Analyzed 92.5 min wall time", "Analyzed 45.2 min wall time")
content = content.replace("Exp 2065", "Exp 2089") # wait, the regex might have already replaced Exp 2065 with Exp 2089

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("HTML card completely updated.")
